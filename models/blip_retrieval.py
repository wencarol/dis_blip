from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import math

from models.blip import create_vit, init_tokenizer, load_checkpoint

class BLIP_Retrieval(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/small6_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   # 生成token的，之后可以在这里改
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(4 * text_width, text_width),
        #     nn.ReLU(),
        #     nn.Linear(text_width, 2),
        # )

        self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        self.negative_all_rank = negative_all_rank
        
        
    def forward(self, image, caption, alpha, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image)  #torch.Size([8, 577, 768])    len(image_q)=6 每个torch.Size([4, 12, 577, 64])  self.visual_encoder.blocks[0].attn.qkv
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        #torch.Size([4, 577])
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    # torch.Size([4, 256])
        image_logit = F.normalize(image_embeds,dim=-1)  #自己加的，取最后一行

        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', output_hidden_states=True, output_attentions=True)            # tuple len=13, torch.Size([4, 35, 768])
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)     # torch.Size([4, 256])
        text_logit = F.normalize(text_output.last_hidden_state, dim=-1)
        
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_m_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_m_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp  
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp   

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_m_all / self.temp 
        sim_t2i = text_feat @ image_feat_m_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        
        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)        


##==================KD==============================
        encoder_input_ids = text.input_ids.clone() # torch.Size([8, 35])
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id #torch.Size([8])
        encoder_input_ids_chunk = torch.repeat_interleave(encoder_input_ids, encoder_input_ids.shape[0], 0).chunk(encoder_input_ids.shape[0])  #tuple len=8 每个torch.Size([4, 35])
        attention_mask_chunk = torch.repeat_interleave(text.attention_mask, encoder_input_ids.shape[0], 0).chunk(encoder_input_ids.shape[0]) #tuple len=8 每个torch.Size([4, 35])
        image_embeds_chunk = image_embeds.repeat(image_embeds.shape[0], 1, 1).chunk(encoder_input_ids.shape[0]) #tuple len=8 每个torch.Size([8, 577, 768])
        image_atts_chunk = image_atts.repeat(image_embeds.shape[0], 1).chunk(encoder_input_ids.shape[0]) #tuple len=8 每个torch.Size([8, 577)

        
        # 自己修改部分
        onetower_list = []
        tmp_loss=0
        with torch.no_grad():
            for i in range(len(encoder_input_ids_chunk)):  #遍历一个batch里的文字
                output_pos_t = self.text_encoder(encoder_input_ids_chunk[i], 
                                                attention_mask = attention_mask_chunk[i],
                                                encoder_hidden_states = image_embeds_chunk[i],  # 8个图片的embedding
                                                encoder_attention_mask = image_atts_chunk[i],      
                                                return_dict = True,
                                                output_hidden_states=True,
                                                output_attentions=True
                                                ) # output_pos_t.last_hidden_state   torch.Size([4, 35, 768])
                # import pdb
                # pdb.set_trace()
                '''BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30524, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (2): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (3): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (4): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (5): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (6): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (7): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (8): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (9): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (10): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (11): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (crossattention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
)'''

                #encoder_outputs = self.encoder(config=med_config)
                #out = model.bert(torch.tensor(input_ids).reshape(1, -1).detach().to('cuda'), token_type_ids=torch.tensor(token_type_ids).reshape(1, -1).detach().to('cuda'))
                #attention = out[2]
                #attention = output_pos_t.hidden_states #13个 每个torch.Size([4, 35, 768])
                attention = output_pos_t[-1] #tuple 13个 每个torch.Size([4, 35, 768])

                #attn_list.append(F.softmax(attention)[:, 1].reshape(1, encoder_input_ids.shape[0]))
                onetower_list.append(F.softmax(self.itm_head(output_pos_t.last_hidden_state[:,0,:]), dim=1)[:, 1].reshape(1, encoder_input_ids.shape[0])) #过softmax的得分加入list

                for j in range(6):
                    # 获取第i层的Query向量
                    Q_text = self.text_encoder.encoder.layer[j].attention.self.query(text_output.last_hidden_state)

                    # 获取第i层的Key向量
                    K_text = self.text_encoder.encoder.layer[j].attention.self.key(text_output.last_hidden_state)

                    # 获取第i层的Value向量
                    V_text = self.text_encoder.encoder.layer[j].attention.self.value(text_output.last_hidden_state)

                    # one_tower
                    Q_one = self.text_encoder.encoder.layer[j].attention.self.query(output_pos_t.last_hidden_state)
                    K_one = self.text_encoder.encoder.layer[j].attention.self.key(output_pos_t.last_hidden_state)
                    V_one = self.text_encoder.encoder.layer[j].attention.self.value(output_pos_t.last_hidden_state)

                    Q_cross = self.text_encoder.encoder.layer[j].crossattention.self.query(output_pos_t.last_hidden_state)
                    K_cross = self.text_encoder.encoder.layer[j].crossattention.self.key(image_embeds_chunk[i])
                    V_cross = self.text_encoder.encoder.layer[j].crossattention.self.value(image_embeds_chunk[i])

                    # image
                    qkv=self.visual_encoder.blocks[j].attn.qkv(image_embeds)  # Linear(in_features=768, out_features=2304, bias=True)   torch.Size([8, 577, 2304])
                    Q_image, K_image, V_image = qkv.chunk(3, dim=-1)  # torch.Size([8, 577, 768])

                    # QK_text = Q_text @ K_text.transpose(-2, -1)
                    # QK_one =  Q_one @ K_one.transpose(-2,-1)
                    # QK_text = F.softmax(QK_text, dim=-1)
                    # QK_one = F.softmax(QK_one, dim=-1)
                    # #tmp_loss += F.kl_div(QK_text.softmax(dim=-1).log(),QK_one.softmax(dim=-1).log(), reduction='sum')
                    # tmp_loss += F.kl_div(QK_text.softmax(dim=-1).log(),QK_one, reduction='mean')

                    # VR_text = V_text @ V_text.transpose(-2, -1)
                    # VR_one =  V_one @ V_one.transpose(-2,-1)
                    # VR_text = F.softmax(VR_text, dim=-1)
                    # VR_one = F.softmax(VR_one, dim=-1)
                    # #tmp_loss += F.kl_div(VR_text.softmax(dim=-1).log(),VR_one.softmax(dim=-1).log(), reduction='sum')
                    # tmp_loss += F.kl_div(VR_text.softmax(dim=-1).log(),VR_one, reduction='mean')

                    QK_text_image = (Q_text @ K_image.transpose(-2, -1))/math.sqrt(768)
                    QK_cross = (Q_cross @ K_cross.transpose(-2, -1) )/math.sqrt(768)
                    QK_cross = QK_cross.masked_fill(~image_atts_chunk[i].bool()[:, None, None, :], float("-inf"))
                    smooth_factor = 1e-12  # 平滑因子
                    QK_text_image = F.softmax(QK_text_image, dim=-1)
                    QK_cross = F.softmax(QK_cross, dim=-1)
                    QK_text_image = (QK_text_image + smooth_factor) / (QK_text_image.sum(dim=-1, keepdim=True) + smooth_factor)
                    QK_cross = (QK_cross + smooth_factor) / (QK_cross.sum(dim=-1, keepdim=True) + smooth_factor)
                    tmp_loss += F.kl_div(F.log_softmax(QK_text_image.float(), dim=-1), F.softmax(QK_cross.float(), dim=-1), reduction='batchmean')
                    #QK_text_image = F.softmax(QK_text_image, dim=-1)
                    '''import pdb
                    pdb.set_trace()'''
                    #QK_cross = F.softmax(QK_cross, dim=-1)
                    #tmp_loss += F.kl_div(F.log_softmax(QK_text_image.float(), dim=-1), F.softmax(QK_cross.float(), dim=-1), reduction='batchmean')

                    '''VR_text_image = V_image @ V_image.transpose(-2, -1)
                    VR_cross =  V_cross @ V_cross.transpose(-2,-1)
                    VR_text_image = F.softmax(VR_text_image, dim=-1)
                    VR_cross = F.softmax(VR_cross, dim=-1)
                    #tmp_loss += F.kl_div(VR_text_image.softmax(dim=-1).log(),VR_cross.softmax(dim=-1), reduction='mean')
                    tmp_loss += F.kl_div(F.log_softmax(VR_text_image.float(), dim=-1), F.softmax(VR_cross.float(), dim=-1), reduction='mean')'''

                # # 获取第i层的Query向量
                # Q_text = self.text_encoder.encoder.layer[5].attention.self.query(text_output.last_hidden_state)

                # # 获取第i层的Key向量
                # K_text = self.text_encoder.encoder.layer[5].attention.self.key(text_output.last_hidden_state)

                # # 获取第i层的Value向量
                # V_text = self.text_encoder.encoder.layer[5].attention.self.value(text_output.last_hidden_state)

                # # one_tower
                # # Q_one = self.text_encoder.encoder.layer[6].attention.self.query(output_pos_t.last_hidden_state)
                # # K_one = self.text_encoder.encoder.layer[6].attention.self.key(output_pos_t.last_hidden_state)
                # # V_one = self.text_encoder.encoder.layer[6].attention.self.value(output_pos_t.last_hidden_state)

                # Q_cross = self.text_encoder.encoder.layer[5].crossattention.self.query(output_pos_t.last_hidden_state)
                # K_cross = self.text_encoder.encoder.layer[5].crossattention.self.key(image_embeds_chunk[i])
                # V_cross = self.text_encoder.encoder.layer[5].crossattention.self.value(image_embeds_chunk[i])

                # # image
                # qkv=self.visual_encoder.blocks[5].attn.qkv(image_embeds)  # Linear(in_features=768, out_features=2304, bias=True)   torch.Size([8, 577, 2304])
                # Q_image, K_image, V_image = qkv.chunk(3, dim=-1)  # torch.Size([8, 577, 768])

                # QK_text_image = (Q_text @ K_image.transpose(-2, -1))/math.sqrt(768)
                # #sim = text_logit @ image_logit.transpose(-2, -1)
                # QK_cross = (Q_cross @ K_cross.transpose(-2, -1) )/math.sqrt(768)
                # QK_cross = QK_cross.masked_fill(~image_atts_chunk[i].bool()[:, None, None, :], float("-inf"))
                # #smooth_factor = 1e-12  # 平滑因子
                # #QK_text_image = F.softmax(QK_text_image, dim=-1)
                # #QK_cross = F.softmax(QK_cross, dim=-1)
                # #sim = F.softmax(sim, dim=-1)
                # # QK_text_image = (QK_text_image + smooth_factor) / (QK_text_image.sum(dim=-1, keepdim=True) + smooth_factor)
                # # QK_cross = (QK_cross + smooth_factor) / (QK_cross.sum(dim=-1, keepdim=True) + smooth_factor)
                # tmp_loss += F.kl_div(F.log_softmax(QK_text_image.float(), dim=-1), F.softmax(QK_cross.float(), dim=-1), reduction='batchmean')
                # #tmp_loss += F.kl_div(F.log_softmax(sim.float(), dim=-1), F.softmax(QK_cross.float(), dim=-1), reduction='batchmean')


        #onetower_attn = torch.cat(attn_list, 0) 
        ranker_score = torch.cat(onetower_list, 0)   
        
        #双流
        #text_attention=text_output[-1]  #tuple 13个 每个torch.Size([4, 35, 768])

        '''# t2t蒸馏
        tmp_loss = 0
        for i in range(6):
            # tmp_loss += F.kl_div(F.softmax(text_attention[i],dim=-1),F.softmax(attention[i],dim=-1),reduction='sum')
            tmp_loss += F.kl_div(text_attention[i].softmax(dim=-1).log(),attention[i].softmax(dim=-1).log(),reduction='sum')'''

        # text_q=[] #len=12, 每个torch.Size([4, 35, 768])
        # text_k=[]
        # text_v=[]
        # tmp_loss=0

        # for j in range(6):
        #     # 获取第i层的Query向量
        #     Q_text = self.text_encoder.encoder.layer[j].attention.self.query(text_output.last_hidden_state)

        #     # 获取第i层的Key向量
        #     K_text = self.text_encoder.encoder.layer[j].attention.self.key(text_output.last_hidden_state)

        #     # 获取第i层的Value向量
        #     V_text = self.text_encoder.encoder.layer[j].attention.self.value(text_output.last_hidden_state)

        #     # one_tower
        #     Q_one = self.text_encoder.encoder.layer[j].attention.self.query(output_pos_t.last_hidden_state)
        #     K_one = self.text_encoder.encoder.layer[j].attention.self.key(output_pos_t.last_hidden_state)
        #     V_one = self.text_encoder.encoder.layer[j].attention.self.value(output_pos_t.last_hidden_state)

        #     Q_cross = self.text_encoder.encoder.layer[j].crossattention.self.query(output_pos_t.last_hidden_state)
        #     K_cross = self.text_encoder.encoder.layer[j].crossattention.self.key(image_embeds_chunk[i])
        #     V_cross = self.text_encoder.encoder.layer[j].crossattention.self.value(image_embeds_chunk[i])

            # # image
            # qkv=self.visual_encoder.blocks[i].attn.qkv(image_embeds)  # Linear(in_features=768, out_features=2304, bias=True)   torch.Size([8, 577, 2304])
            # Q_image, K_image, V_image = qkv.chunk(3, dim=-1)  # torch.Size([8, 577, 768])




        #     QK_text = Q_text @ K_text.transpose(-2, -1)
        #     QK_one =  Q_one @ K_one.transpose(-2,-1)
        #     QK_text = F.softmax(QK_text, dim=-1)
        #     QK_one = F.softmax(QK_one, dim=-1)
        #     #tmp_loss += F.kl_div(QK_text.softmax(dim=-1).log(),QK_one.softmax(dim=-1).log(), reduction='sum')
        #     tmp_loss += F.kl_div(QK_text.softmax(dim=-1).log(),QK_one, reduction='mean')

        #     VR_text = V_text @ V_text.transpose(-2, -1)
        #     VR_one =  V_one @ V_one.transpose(-2,-1)
        #     VR_text = F.softmax(VR_text, dim=-1)
        #     VR_one = F.softmax(VR_one, dim=-1)
        #     #tmp_loss += F.kl_div(VR_text.softmax(dim=-1).log(),VR_one.softmax(dim=-1).log(), reduction='sum')
        #     tmp_loss += F.kl_div(VR_text.softmax(dim=-1).log(),VR_one, reduction='mean')
 
            # text_q.append(Q)
            # text_k.append(K)
            # text_v.append(V)

            # 获取第一层的Attention权重
            # attention_weights = model.encoder.layer[0].attention(outputs.last_hidden_state, attention_mask)[0]


        '''onetower_q=[]
        onetower_k=[]
        onetower_v=[]
        for i in range(6):
            Q = output_pos_t.encoder.layer[i].attention.self.query(output_pos_t.last_hidden_state)
            K = output_pos_t.encoder.layer[i].attention.self.key(output_pos_t.last_hidden_state)
            V = output_pos_t.encoder.layer[i].attention.self.value(output_pos_t.last_hidden_state)

            onetower_q.append(Q)
            onetower_k.append(K)
            onetower_v.append(V)'''
        

            
            

#         for i in range(6):
#             import pdb
#             pdb.set_trace()
#             qkv=self.visual_encoder.blocks[i].attn.qkv(image_embeds)  # Linear(in_features=768, out_features=2304, bias=True)   torch.Size([8, 577, 2304])
#             q, k, v = qkv.chunk(3, dim=-1)
#             """VisionTransformer(
#   (patch_embed): PatchEmbed(
#     (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
#     (norm): Identity()
#   )
#   (pos_drop): Dropout(p=0.0, inplace=False)
#   (blocks): ModuleList(
#     (0): Block(
#       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (attn): Attention(
#         (qkv): Linear(in_features=768, out_features=2304, bias=True)
#         (attn_drop): Dropout(p=0.0, inplace=False)
#         (proj): Linear(in_features=768, out_features=768, bias=True)
#         (proj_drop): Dropout(p=0.0, inplace=False)
#       )
#       (drop_path): Identity()
#       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (mlp): Mlp(
#         (fc1): Linear(in_features=768, out_features=3072, bias=True)
#         (act): GELU()
#         (fc2): Linear(in_features=3072, out_features=768, bias=True)
#         (drop): Dropout(p=0.0, inplace=False)
#       )
#     )
#     (1): Block(
#       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (attn): Attention(
#         (qkv): Linear(in_features=768, out_features=2304, bias=True)
#         (attn_drop): Dropout(p=0.0, inplace=False)
#         (proj): Linear(in_features=768, out_features=768, bias=True)
#         (proj_drop): Dropout(p=0.0, inplace=False)
#       )
#       (drop_path): Identity()
#       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (mlp): Mlp(
#         (fc1): Linear(in_features=768, out_features=3072, bias=True)
#         (act): GELU()
#         (fc2): Linear(in_features=3072, out_features=768, bias=True)
#         (drop): Dropout(p=0.0, inplace=False)
#       )
#     )
#     (2): Block(
#       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (attn): Attention(
#         (qkv): Linear(in_features=768, out_features=2304, bias=True)
#         (attn_drop): Dropout(p=0.0, inplace=False)
#         (proj): Linear(in_features=768, out_features=768, bias=True)
#         (proj_drop): Dropout(p=0.0, inplace=False)
#       )
#       (drop_path): Identity()
#       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (mlp): Mlp(
#         (fc1): Linear(in_features=768, out_features=3072, bias=True)
#         (act): GELU()
#         (fc2): Linear(in_features=3072, out_features=768, bias=True)
#         (drop): Dropout(p=0.0, inplace=False)
#       )
#     )
#     (3): Block(
#       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (attn): Attention(
#         (qkv): Linear(in_features=768, out_features=2304, bias=True)
#         (attn_drop): Dropout(p=0.0, inplace=False)
#         (proj): Linear(in_features=768, out_features=768, bias=True)
#         (proj_drop): Dropout(p=0.0, inplace=False)
#       )
#       (drop_path): Identity()
#       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (mlp): Mlp(
#         (fc1): Linear(in_features=768, out_features=3072, bias=True)
#         (act): GELU()
#         (fc2): Linear(in_features=3072, out_features=768, bias=True)
#         (drop): Dropout(p=0.0, inplace=False)
#       )
#     )
#     (4): Block(
#       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (attn): Attention(
#         (qkv): Linear(in_features=768, out_features=2304, bias=True)
#         (attn_drop): Dropout(p=0.0, inplace=False)
#         (proj): Linear(in_features=768, out_features=768, bias=True)
#         (proj_drop): Dropout(p=0.0, inplace=False)
#       )
#       (drop_path): Identity()
#       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (mlp): Mlp(
#         (fc1): Linear(in_features=768, out_features=3072, bias=True)
#         (act): GELU()
#         (fc2): Linear(in_features=3072, out_features=768, bias=True)
#         (drop): Dropout(p=0.0, inplace=False)
#       )
#     )
#     (5): Block(
#       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (attn): Attention(
#         (qkv): Linear(in_features=768, out_features=2304, bias=True)
#         (attn_drop): Dropout(p=0.0, inplace=False)
#         (proj): Linear(in_features=768, out_features=768, bias=True)
#         (proj_drop): Dropout(p=0.0, inplace=False)
#       )
#       (drop_path): Identity()
#       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#       (mlp): Mlp(
#         (fc1): Linear(in_features=768, out_features=3072, bias=True)
#         (act): GELU()
#         (fc2): Linear(in_features=3072, out_features=768, bias=True)
#         (drop): Dropout(p=0.0, inplace=False)
#       )
#     )
#   )
#   (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
# )"""

        #image_attention=self.visual_encoder

        #interaction
        interaction_text_to_image = F.softmax(torch.matmul(text_logit, image_logit.transpose(1, 2)), dim=-1)
        interaction_image_to_text = F.softmax(torch.matmul(image_logit, text_logit.transpose(1, 2)), dim=-1)
        u = torch.matmul(interaction_text_to_image, image_logit)  # Attention-based pooling for X
        v = torch.matmul(interaction_image_to_text, text_logit)  # Attention-based pooling for Y

        image_feat_vir = F.normalize(self.vision_proj(u[:,0,:]),dim=-1)
        text_feat_vir = F.normalize(self.text_proj(v[:,0,:]),dim=-1)


        # # r = torch.cat([u, v, u - v, torch.max(u, v)], dim=-1)  # Concatenate pooled representations
        # # y_pred = self.mlp(r)  # MLP prediction

        # # loss = self.criterion(y_pred, labels)  # Calculate loss

        # # maxsimilarity
        # maxsim_score = torch.zeros((8, 8)).to(text_logit.device)
        # for i in range(1, 35):
        #     sim_tensor = u[:,i,:] @ v[:,1:577,:].transpose(1, 2)
        #     sim_score, _ = torch.max(sim_tensor, dim=2)
        #     maxsim_score += sim_score.t()


        # # maxsimilarity
        # maxsim_score = torch.zeros((8, 8)).to(text_logit.device)
        # for i in range(8):
        #     sim_tensor = text_logit[i,1:35,:] @ image_logit[:,1:577,:].transpose(1, 2)
        #     sim_score, _ = torch.max(sim_tensor, dim=2)
        #     sim_score = torch.sum(sim_score, dim=1)
        #     maxsim_score[i] = sim_score.t()
        # for i in range(1, 35):
        #     sim_tensor = text_logit[:,i,:] @ image_logit[:,1:577,:].transpose(1, 2)
        #     sim_score, _ = torch.max(sim_tensor, dim=2)
        #     maxsim_score += sim_score.t()

        

        #sim_t2i_kd = text_feat @ image_feat.t() #/ self.temp
        sim_t2i_kd = text_feat_vir @ image_feat_vir.t()

        loss_ita += tmp_loss
        #loss_ita += F.kl_div(ranker_score.softmax(dim=-1).log(),maxsim_score.softmax(dim=-1),reduction='sum')  # KL散度
        #loss_ita += F.kl_div(maxsim_score.softmax(dim=-1).log(),ranker_score.softmax(dim=-1),reduction='sum')
        #loss_ita += torch.sum((sim_t2i_kd - ranker_score)**2, dim=1).mean() # mse
        loss_ita += F.kl_div(sim_t2i_kd.softmax(dim=-1).log(),ranker_score.softmax(dim=-1),reduction='sum')

        #原代码
        '''onetower_list = []

        with torch.no_grad():
            for i in range(len(encoder_input_ids_chunk)):  #遍历一个batch里的文字
                output_pos_t = self.text_encoder(encoder_input_ids_chunk[i], #遍历每一个文字
                                            attention_mask = attention_mask_chunk[i],
                                            encoder_hidden_states = image_embeds_chunk[i],  # 64个图片的embedding
                                            encoder_attention_mask = image_atts_chunk[i],      
                                            return_dict = True,
                                            ) 

                onetower_list.append(F.softmax(self.itm_head(output_pos_t.last_hidden_state[:,0,:]), dim=1)[:, 1].reshape(1, encoder_input_ids.shape[0])) #过softmax的得分加入list
        ranker_score = torch.cat(onetower_list, 0)   
        ranker_score = ranker_score #- torch.mean(ranker_score, dim=1).unsqueeze(1)     单流模型输出的                
        

        sim_t2i_kd = text_feat @ image_feat.t() #/ self.temp    双流模型 图片和文字embedding的内积 n个文字和n个图片的任意pair的相似度

        loss_ita += F.kl_div(ranker_score.softmax(dim=-1).log(),sim_t2i_kd.softmax(dim=-1),reduction='sum')  #KL散度
        #loss_ita += torch.sum((sim_t2i_kd - ranker_score)**2, dim=1).mean()   #mse '''


##-----------------------------------------------
        # encoder_input_ids = text.input_ids.clone()
        # encoder_input_ids[:,0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )  
        
        if self.negative_all_rank:    
            # compute sample similarity
            with torch.no_grad():                
                mask = torch.eq(idx, idxs.t())

                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = image_feat @ text_feat_world.t() / self.temp 
                sim_t2i = text_feat @ image_feat_world.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            image_embeds_world = all_gather_with_grad(image_embeds) 

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from all ranks) for each image
            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text.attention_mask)        

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])
                
        else:
            with torch.no_grad():                
                mask = torch.eq(idx, idx.t())
                
                sim_i2t = image_feat @ text_feat.t() / self.temp 
                sim_t2i = text_feat @ image_feat.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            # select a negative image (from same rank) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from same rank) for each image    
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])            
            
        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,      
                                       return_dict = True,
                                      )                         
          

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     

        return loss_ita, loss_itm 
 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  


def blip_retrieval(pretrained='',**kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
