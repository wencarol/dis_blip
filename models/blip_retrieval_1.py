from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint

class BLIP_Retrieval(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
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
        
        image_embeds, image_attention = self.visual_encoder(image)  #torch.Size([4, 577, 768])     image_attention:list 6个 每个torch.Size([4, 577, 768])
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        #torch.Size([4, 577])
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    # torch.Size([4, 256])

        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', output_hidden_states=True)            # tuple len=13, torch.Size([4, 35, 768])
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)     # torch.Size([4, 256])
        
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m, _ = self.visual_encoder_m(image) 
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
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id
        encoder_input_ids_chunk = torch.repeat_interleave(encoder_input_ids, encoder_input_ids.shape[0], 0).chunk(encoder_input_ids.shape[0])
        attention_mask_chunk = torch.repeat_interleave(text.attention_mask, encoder_input_ids.shape[0], 0).chunk(encoder_input_ids.shape[0])
        image_embeds_chunk = image_embeds.repeat(image_embeds.shape[0], 1, 1).chunk(encoder_input_ids.shape[0])
        image_atts_chunk = image_atts.repeat(image_embeds.shape[0], 1).chunk(encoder_input_ids.shape[0])

        onetower_list = []
        with torch.no_grad():
            for i in range(len(encoder_input_ids_chunk)):  #遍历一个batch里的文字
                output_pos_t = self.text_encoder(encoder_input_ids_chunk[i], 
                                            attention_mask = attention_mask_chunk[i],
                                            encoder_hidden_states = image_embeds_chunk[i],  # 64个图片的embedding
                                            encoder_attention_mask = image_atts_chunk[i],      
                                            return_dict = True,
                                            output_hidden_states=True
                                            ) # output_pos_t.last_hidden_state   torch.Size([4, 35, 768])
                #encoder_outputs =  self.encoder(config=med_config)
                #out = model.bert(torch.tensor(input_ids).reshape(1, -1).detach().to('cuda'), token_type_ids=torch.tensor(token_type_ids).reshape(1, -1).detach().to('cuda'))
                #attention = out[2]
                attention = output_pos_t.hidden_states #13个 每个torch.Size([4, 35, 768])


                #attn_list.append(F.softmax(attention)[:, 1].reshape(1, encoder_input_ids.shape[0]))
                onetower_list.append(F.softmax(self.itm_head(output_pos_t.last_hidden_state[:,0,:]), dim=1)[:, 1].reshape(1, encoder_input_ids.shape[0])) #过softmax的得分加入list
        #onetower_attn = torch.cat(attn_list, 0) 
        
        #双流
        text_attention=text_output.hidden_states      #13个 torch.Size([4, 35, 768])
        

        #image_attention=self.visual_encoder
        import pdb
        pdb.set_trace()


        #loss_ita += F.kl_div(onetower_attn.softmax(dim=-1).log(),image_atts.softmax(dim=-1),reduction='sum')
        #loss_ita += torch.sum((sim_t2i_kd - ranker_score)**2, dim=1).mean()

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

        loss_ita += F.kl_div(ranker_score.softmax(dim=-1).log(),sim_t2i_kd.softmax(dim=-1),reduction='sum')
        #loss_ita += torch.sum((sim_t2i_kd - ranker_score)**2, dim=1).mean()'''


##-----------------------------------------------


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
