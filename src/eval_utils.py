# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from functools import partial
from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
import pickle

from utils import is_master


def build_bidirectional_fashion_prompts(relative_captions, prompt_template="a photo of * that {first} and {second}"):
    arr = np.array(relative_captions, dtype=object)
    if arr.ndim == 2 and arr.shape[0] == 2:
        caption_pairs = arr.T.tolist()
    else:
        caption_pairs = arr.tolist()

    prompts = []
    prompts_reversed = []
    for pair in caption_pairs:
        if len(pair) != 2:
            raise ValueError(f"Expected caption pair of length 2, got: {pair}")
        cap1 = str(pair[0]).strip(".?, ")
        cap2 = str(pair[1]).strip(".?, ")
        prompts.append(prompt_template.format(first=cap1, second=cap2))
        prompts_reversed.append(prompt_template.format(first=cap2, second=cap1))
    return prompts, prompts_reversed


def prepare_img(img_file, transform):
    return transform(Image.open(img_file))

def visualize_results(model, img2text, args, prompt, dataloader):        
    model.eval()
    img2text.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    for p in prompt:
        text_tokens = tokenize(p)
        text.append(text_tokens)
        assert id_split in text_tokens
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            dict_save = {}
            dict_save['feats'] = all_image_features.data.cpu().numpy()
            dict_save['path'] = all_image_filenames
            with open(path_save,"wb") as f:
                pickle.dump(dict_save,f)
    f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    for query in query_file.split(","):        
        logging.info("retrieve image of {}".format(query))
        transform = _transform(model.visual.input_resolution)
        query_img = prepare_img(query, transform)
        query_img = torch.unsqueeze(query_img, 0)    
        query_img = query_img.cuda(args.gpu, non_blocking=True)
        img_feature = m.encode_image(query_img) 
        query_img_feature = img2text(img_feature)
        composed_feature = m.encode_text_img_vis(text, query_img_feature, split_ind=id_split)
        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
        text_feature = m.encode_text(text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = composed_feature @ all_image_features.T
        _, indices = torch.sort(similarity, descending=True)        
        logging.info("Composed feature result")
        for i, caption in enumerate(prompt):
            logging.info("for prompt {}".format(caption))
            for j, ind in enumerate(indices[i][:8]):
                logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
        image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:8])] 
                        for i, caption in enumerate(prompt)]
        html_txt += make_html(prompt, query, image_paths, args.demo_out)
    f.write(html_txt)

def make_html(prompts, query_image, images, path_html):
    import shutil
    html_all = """"""        
    for i in range(len(prompts)):
        prompt = prompts[i]            
        query_image_local = os.path.join(path_html, "images", query_image.split("/")[-1])
        query_image_local_path = os.path.join("images", query_image.split("/")[-1])
        shutil.copy(query_image, query_image_local)
        image_list = images[i]        
        html = """<table><tr>"""    
        html += """<td><p style="display:inline-block;vertical-align;font-size:20px">%s</p></td>"""%(prompt)
        html += """<td><p style="margin-right: 50px;"><img src="%s" height="100"></p></td>"""%(query_image_local_path)
        for image in image_list:
            image_local = os.path.join(path_html, "images", image.split("/")[-1])
            image_path = os.path.join("images", image.split("/")[-1])
            shutil.copy(image, image_local)
            html += """<td><img src="%s" height=%s></td>"""%(image_path, 200)
        html += """</tr></table>"""
        html_all += html
    return html_all
    #f.write(html_all)


def evaluate_imgnet_retrieval(model, img2text, args, prompt, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_image_features = []  
    all_target_labels = []      
    m = model.module if args.distributed or args.dp else model
    n_class = 1000
   
    with torch.no_grad():
        ## Extract target image features. 
        for batch in tqdm(target_loader):
            images, labels = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            all_target_labels.append(labels)
            logit_scale = m.logit_scale.exp()
            logit_scale = logit_scale.mean()   

        ## Extract query features 
        for p_ind, p in enumerate(prompt):            
            ## which token has to be replaced with image features
            id_split = tokenize(["*"])[0][1]
            text = tokenize(p).view(1, -1)
            text = text.cuda(args.gpu, non_blocking=True)
            ## text only features (domain name only)
            text_only = p.replace("*", "")
            text_only = tokenize(text_only).view(1, -1)            
            text_only = text_only.cuda(args.gpu, non_blocking=True)                        
            text_only_features = m.encode_text(text_only)
            text_only_features_normed = text_only_features / text_only_features.norm(dim=-1, keepdim=True)

            all_query_features = []
            all_query_image_features = []
            all_query_mixture_features = []
            all_query_labels = []
            all_text_features = []
            for batch in tqdm(query_loader):
                images, labels = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    labels = labels.cuda(args.gpu, non_blocking=True)
                ## Label is decided by class label and images' domain
                labels += n_class * p_ind
                image_features = m.encode_image(images)
                 ## Composed feature extraction
                image_features_query = img2text(image_features)                      
                composed_feature = m.encode_text_img_retrieval(text, image_features_query, split_ind=id_split)                            
                composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
                ## Image feature only
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
                ## average of image and text features
                mixture_features = image_features + text_only_features_normed
                mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)       

                all_text_features.append(text_only_features_normed.repeat((image_features.shape[0], 1)))
                all_query_features.append(composed_feature)
                all_query_image_features.append(image_features)
                all_query_mixture_features.append(mixture_features)
                all_query_labels.append(labels)

            metric_func = partial(get_metrics_imgnet, 
                image_features=torch.cat(all_image_features), 
                query_labels=torch.cat(all_query_labels),
                target_labels=torch.cat(all_target_labels),
                )

            feats = {'composed': torch.cat(all_query_features), 
                    'image': torch.cat(all_query_image_features),
                    'text': torch.cat(all_text_features),
                    'mixture': torch.cat(all_query_mixture_features)}        

            for key, value in feats.items():
                metrics = metric_func(query_features=value)
                logging.info(
                f"Eval {key} Feature"
                + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_coco(model, img2text, args, loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_mixture_features = []  
    all_composed_features_with_class = []  
    all_text_full_features = [] 

    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()
    with torch.no_grad():
        for batch in tqdm(loader):
            images, region_images, text_full, text_with_blank, text_with_blank_query, filename, raw_text = batch            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                region_images = region_images.cuda(args.gpu, non_blocking=True)
                text_full = text_full.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                text_with_blank_query = text_with_blank_query.cuda(args.gpu, non_blocking=True)

            ## Target image features 
            image_features = m.encode_image(images)             
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
            id_split = tokenize(["*"])[0][1]
            ## Composed image features
            query_image_features = m.encode_image(region_images)
            query_image_tokens = img2text(query_image_features)          
            composed_feature_with_class = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens, split_ind=id_split, repeat=False)                        
            composed_feature_with_class = composed_feature_with_class / composed_feature_with_class.norm(dim=-1, keepdim=True)        
            ## Text only features
            text_full_features = m.encode_text(text_full)
            text_full_features = text_full_features / text_full_features.norm(dim=-1, keepdim=True)            
            ## Query only features
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)                               
            ## Mixed featurs
            mixture_features = query_image_features + text_full_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)            

            all_image_features.append(image_features.cpu())
            all_text_full_features.append(text_full_features.cpu())       
            all_query_image_features.append(query_image_features.cpu())
            all_mixture_features.append(mixture_features.cpu())                        
            all_composed_features_with_class.append(composed_feature_with_class.cpu())            

        metric_func = partial(get_metrics_coco, 
                image_features=torch.cat(all_image_features), 
                logit_scale=logit_scale
                )
        feats = {'composed': torch.cat(all_composed_features_with_class), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_text_full_features),
                 'mixture': torch.cat(all_mixture_features)}        

        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_cirr(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_raw_captions = []
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        all_group_members = []
        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, answer_paths, raw_captions, group_members = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for path in ref_paths:
                all_ref_paths.append(path)
            for path in answer_paths:
                all_answer_paths.append(path)
            for cap in raw_captions:
                all_raw_captions.append(cap)
            # group_members from DataLoader: tuple of N_members lists, each of batch_size strings
            gm = np.array(group_members).T.tolist()  # -> [batch_size, N_members]
            all_group_members.extend(gm)

            caption_features = m.encode_text(caption_only)
            ## Composed features
            query_image_features = m.encode_image(ref_images)
            query_image_tokens = img2text(query_image_features)
            composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)                

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features            
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)                        

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        
        def _norm_name(x):
            x = os.path.basename(str(x))
            if not x.endswith(".png"):
                x = x + ".png"
            return x

        all_target_paths = np.array([_norm_name(x) for x in all_target_paths])
        all_ref_paths    = np.array([_norm_name(x) for x in all_ref_paths])
        all_answer_paths = np.array([_norm_name(x) for x in all_answer_paths])
        all_group_members = [[_norm_name(m) for m in members] for members in all_group_members]
        
        metric_func = partial(get_metrics_cirr, 
                image_features=torch.cat(all_image_features), 
                reference_names=all_ref_paths, 
                index_names=all_target_paths, 
                target_names=all_answer_paths,
                group_members=all_group_members)

        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        all_metrics = {}
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            all_metrics[key] = metrics
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return all_metrics


def evaluate_cirr_test(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_ids = []
    all_group_members = []

    m = model.module if args.distributed or args.dp else model   
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, pairids, text_with_blank_raw, group_members = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for ids in pairids:
                all_ids.append(ids)
            for path in ref_paths:
                all_ref_paths.append(path)
            gm = np.array(group_members).T.tolist()
            all_group_members.extend(gm)

            caption_features = m.encode_text(caption_only)
            query_image_features = m.encode_image(ref_images)

            if getattr(args, 'eval_combiner', False):
                composed_feature = img2text(query_image_features, caption_features)
            else:
                query_image_tokens = img2text(query_image_features)
                composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)            

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        res_all = {}
        metrics_func = partial(get_cirr_testoutput, 
                               image_features=torch.cat(all_image_features),
                               reference_names=all_ref_paths,
                               index_names=all_target_paths,
                               id_names=all_ids,
                               group_members=all_group_members)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}        
        for key, value in feats.items():
            res_all[key] = metrics_func(ref_features=value)
    return res_all


def evaluate_fashion(model, img2text, args, source_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_target_paths = []
    all_answer_paths = []
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_caption_features = []  
    all_mixture_features = []  
    all_reference_names = []
    all_captions = []     
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                target_images = target_images.cuda(args.gpu, non_blocking=True)
                target_caption = target_caption.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            query_image_features = m.encode_image(ref_images)
            id_split = tokenize(["*"])[0][1]
            input_captions, input_captions_reversed = build_bidirectional_fashion_prompts(captions)
            tokenized_input_captions = tokenize(input_captions).cuda(args.gpu, non_blocking=True)
            tokenized_input_captions_reversed = tokenize(input_captions_reversed).cuda(args.gpu, non_blocking=True)

            caption_features_forward = m.encode_text(tokenized_input_captions)
            caption_features_reversed = m.encode_text(tokenized_input_captions_reversed)
            caption_features = F.normalize(
                (F.normalize(caption_features_forward, dim=-1) + F.normalize(caption_features_reversed, dim=-1)) / 2,
                dim=-1,
            )
            query_image_tokens = img2text(query_image_features)
            composed_feature_forward = m.encode_text_img_retrieval(
                tokenized_input_captions, query_image_tokens, split_ind=id_split, repeat=False
            )
            composed_feature_reversed = m.encode_text_img_retrieval(
                tokenized_input_captions_reversed, query_image_tokens, split_ind=id_split, repeat=False
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature_raw = F.normalize(
                (F.normalize(composed_feature_forward, dim=-1) + F.normalize(composed_feature_reversed, dim=-1)) / 2,
                dim=-1,
            )
            composed_feature = composed_feature_raw

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)
            all_mixture_features.append(mixture_features)

        metric_func = partial(get_metrics_fashion, 
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths, answer_names=all_answer_paths)
        feats = {'composed': torch.cat(all_composed_features),
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        metrics_by_feature = {}
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            metrics_by_feature[key] = {metric_name: float(metric_val) for metric_name, metric_val in metrics.items()}
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics_by_feature


def get_metrics_coco(image_features, ref_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale.cpu() * image_features @ ref_features.t()).detach().cpu()
    logits_per_ref = logits_per_image.t().detach().cpu()
    logits = {"image_to_ref": logits_per_image, "ref_to_image": logits_per_ref}
    ground_truth = torch.arange(len(ref_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10, 50, 100]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def get_metrics_fashion(image_features, ref_features, target_names, answer_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T    
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())
    # Compute the metrics
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics


def get_metrics_cirr(image_features, ref_features, reference_names, index_names, target_names, group_members=None):
    metrics = {}

    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    num_q = sorted_index_names.shape[0]
    num_g = len(index_names)

    ref_mat = np.repeat(np.array(reference_names), num_g).reshape(num_q, num_g)
    reference_mask = torch.tensor(sorted_index_names != ref_mat)
    flat = sorted_index_names[reference_mask]
    num_g_kept = flat.size // num_q
    sorted_index_names = flat.reshape(num_q, num_g_kept)

    tgt_mat = np.repeat(np.array(target_names), num_g_kept).reshape(num_q, num_g_kept)
    labels = torch.tensor(sorted_index_names == tgt_mat)

    assert torch.equal(
        torch.sum(labels, dim=-1).int(),
        torch.ones(num_q).int()
    ), "Some queries do not have exactly one matching target in gallery."

    for k in [1, 5, 10, 50, 100]:
        k_eff = min(k, num_g_kept)
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k_eff]) / num_q).item() * 100

    # R_subset@K: recall within subset members only
    if group_members is not None:
        group_members_arr = np.array(group_members)  # [num_q, N_members]
        group_mask = (sorted_index_names[..., None] == group_members_arr[:, None, :]).sum(-1).astype(bool)
        sorted_group_names = sorted_index_names[group_mask].reshape(num_q, -1)

        tgt_arr = np.array(target_names)
        for k in [1, 2, 3]:
            k_eff = min(k, sorted_group_names.shape[1])
            hits = (sorted_group_names[:, :k_eff] == tgt_arr[:, None]).any(axis=1)
            metrics[f"R_subset@{k}"] = float(hits.sum()) / num_q * 100

    return metrics



def get_cirr_testoutput(image_features, ref_features, reference_names, index_names, id_names, group_members=None):
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Full-gallery recall dict
    result_dict = {"version": "rc2", "metric": "recall"}
    for ind in range(len(id_names)):
        pairid = str(id_names[ind].item())
        result_dict[pairid] = [sorted_index_names[ind][t].replace(".png", "") for t in range(50)]

    # Subset recall dict
    subset_dict = {"version": "rc2", "metric": "recall_subset"}
    if group_members is not None:
        group_members_arr = np.array(group_members)  # [num_q, N_members]
        group_mask = (sorted_index_names[..., None] == group_members_arr[:, None, :]).sum(-1).astype(bool)
        sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
        for ind in range(len(id_names)):
            pairid = str(id_names[ind].item())
            subset_dict[pairid] = [sorted_group_names[ind][t].replace(".png", "") for t in range(min(3, sorted_group_names.shape[1]))]

    return {"recall": result_dict, "recall_subset": subset_dict}


def get_metrics_imgnet(query_features, image_features, query_labels, target_labels):
    metrics = {}
    num_classes = 7000
    query_onehot = F.one_hot(query_labels, num_classes=num_classes).float()
    target_onehot = F.one_hot(target_labels, num_classes=num_classes).float()
    batches = [(query_features[x:x+100], query_onehot[x:x+100]) for x in range(0, len(query_features), 100)]
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] = 0
        metrics[f"Real2Sketch_P@{k}"] = 0
    for batch in batches:
        feats, labels = batch[0], batch[1]
        logits_per_query = (feats @ image_features.t()).detach().cpu()
        label_matrix = (labels @ target_onehot.t()).detach().cpu()                
        ranking = torch.argsort(logits_per_query, descending=True)
        for k in [1, 5, 10, 50, 100, 200]:
            matrix_k = torch.zeros_like(label_matrix)
            rank_k = ranking[:, :k]
            matrix_k[torch.arange(matrix_k.size(0)).unsqueeze(1), rank_k] = 1
            consistency = matrix_k * label_matrix
            num_correct = torch.sum(consistency, dim=1)
            num_predicted = torch.sum(matrix_k, dim=1)            
            num_total = torch.sum(label_matrix, dim=1)
            recall = torch.mean(num_correct / (num_total+1e-5))
            precision = torch.mean(num_correct / num_predicted)
            metrics[f"Real2Sketch_R@{k}"] += recall * len(feats)
            metrics[f"Real2Sketch_P@{k}"] += precision * len(feats)
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] /= len(query_features)
        metrics[f"Real2Sketch_P@{k}"] /= len(query_features)
    return metrics


import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import logging
import os
from tqdm import tqdm
import numpy as np

from data import CIRCODataset

base_path = Path(__file__).absolute().parents[1].absolute()  # Getting the path to the base directory


def compute_metrics(data_path: Path, predictions_dict: Dict[int, List[int]], ranks: List[int]) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[str, float]]:
    """Computes the Average Precision (AP) and Recall for a given set of predictions.

    Args:
        data_path (Path): Path where the CIRCO datasset is located
        predictions_dict (Dict[int, List[int]]): Predictions of image ids for each query id
        ranks (List[int]): Ranks to consider in the evaluation (e.g., [5, 10, 20])

    Returns:
        Tuple[Dict[int, float], Dict[int, float], Dict[str, float]]: Dictionaries with the AP and Recall for each rank,
            and the semantic mAP@10 for each semantic aspect
    """

    relative_val_dataset = CIRCODataset(data_path, split='val', mode='relative', preprocess=None)

    semantic_aspects_list = ['cardinality', 'addition', 'negation', 'direct_addressing', 'compare_change',
                              'comparative_statement', 'statement_with_conjunction', 'spatial_relations_background',
                              'viewpoint']

    # Initialize empty dictionaries to store the AP and Recall values for each rank
    aps_atk = defaultdict(list)
    recalls_atk = defaultdict(list)
    semantic_aps_at10 = defaultdict(list)

    # Iterate through each query id and its corresponding predictions
    for query_id, predictions in predictions_dict.items():
        target = relative_val_dataset.get_target_img_ids(int(query_id))
        semantic_aspects = relative_val_dataset.get_semantic_aspects(int(query_id))
        gt_img_ids = target['gt_img_ids']
        target_img_id = target['target_img_id']

        # Check if the predictions are unique
        if len(set(predictions)) != len(predictions):
            raise ValueError(f"Query {query_id} has duplicate predictions. Please ensure to provide unique predictions"
                             f"for each query.")

        # gt_img_ids = np.trim_zeros(gt_img_ids)  # remove trailing zeros added for collate_fn (when using dataloader)

        predictions = np.array(predictions, dtype=int)
        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = np.cumsum(ap_labels, axis=0) * ap_labels  # Consider only positions corresponding to GTs
        precisions = precisions / np.arange(1, ap_labels.shape[0] + 1)  # Compute precision for each position

        # Compute the AP and Recall for the given ranks
        for rank in ranks:
            aps_atk[rank].append(float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank)))

        recall_labels = (predictions == target_img_id)
        for rank in ranks:
            recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

        # Compute the AP@10 for each semantic aspect
        for aspect in semantic_aspects:
            semantic_aps_at10[aspect].append(float(np.sum(precisions[:10]) / min(len(gt_img_ids), 10)))

    # Compute the mean AP and Recall for each rank and store them in a dictionary
    map_atk = {}
    recall_atk = {}
    semantic_map_at10 = {}
    for rank in ranks:
        map_atk[rank] = float(np.mean(aps_atk[rank]))
        recall_atk[rank] = float(np.mean(recalls_atk[rank]))

    # Compute the mean AP@10 for each semantic aspect and store them in a dictionary
    for aspect in semantic_aspects_list:
        semantic_map_at10[aspect] = float(np.mean(semantic_aps_at10[aspect]))

    return map_atk, recall_atk, semantic_map_at10



import torch
import torch.nn.functional as F
import logging
import os
import json
import numpy as np
from tqdm import tqdm
from third_party.open_clip.clip import tokenize

def evaluate_circo(model, img2text, args, query_loader, gallery_loader):
    model.eval()
    img2text.eval()
    
    # ---- 0) 检测占位符 '*' 的 token id ----
    # 我们需要找到 '*' 在 tokenizer 中的 ID，以便告诉模型把图像特征插入到哪里
    star_tokens = tokenize(["*"])
    #通常结构是 [SOS, *, EOS, PAD...]，所以取索引 1
    placeholder_token_id = star_tokens[0][1].item() 
    logging.info(f"Detected placeholder '*' token id: {placeholder_token_id}")

    # ---- 1) 提取 Gallery 特征 ----
    logging.info(f"Extracting features for {len(gallery_loader.dataset)} gallery images...")
    gallery_features = []
    gallery_img_ids = []
    
    with torch.no_grad():
        for images, paths in tqdm(gallery_loader, desc="Gallery"):
            images = images.cuda(args.gpu, non_blocking=True)
            
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            gallery_features.append(image_features)
            
            for p in paths:
                filename = os.path.basename(p)
                img_id = int(filename.split('.')[0])
                gallery_img_ids.append(img_id)
                
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_img_ids = np.array(gallery_img_ids)

    # ---- 2) 提取 Query 特征并检索 ----
    logging.info(f"Extracting features for {len(query_loader.dataset)} queries...")
    predictions_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(query_loader, desc="Query"):
            # 适配字典格式
            ref_imgs = batch['reference_img']
            relative_caps = batch['relative_caption']
            query_ids = batch['query_id']
            
            ref_imgs = ref_imgs.cuda(args.gpu, non_blocking=True)

            # 构造 Prompt
            prompts = [f"a photo of * that {cap}" for cap in relative_caps]
            texts = tokenize(prompts).cuda(args.gpu, non_blocking=True)

            # Image -> Soft Token
            ref_img_feats = model.encode_image(ref_imgs)
            soft_tokens = img2text(ref_img_feats)
            
            # ★ 关键修改 ★
            # 1. 使用 encode_text_img_retrieval (它支持 split_ind)
            # 2. 传入 split_ind=placeholder_token_id
            # 3. 传入 repeat=False (因为 texts 已经是 batch 大小了，不需要复制)
            query_feats = model.encode_text_img_retrieval(
                texts, 
                soft_tokens, 
                split_ind=placeholder_token_id, 
                repeat=False 
            )
            
            query_feats = query_feats / query_feats.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            sim_matrix = query_feats @ gallery_features.t()
            
            # 获取 Top 50
            k = 50
            _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
            topk_indices = topk_indices.cpu().numpy()
            
            for i, q_id in enumerate(query_ids):
                # 兼容 q_id 可能是 tensor 的情况
                if isinstance(q_id, torch.Tensor):
                    q_id = q_id.item()
                q_id_int = int(q_id)
                
                retrieved_ids = gallery_img_ids[topk_indices[i]].tolist()
                predictions_dict[q_id_int] = retrieved_ids

    # ---- 3) 保存提交文件 ----
    save_dir = os.path.join(args.logs, args.name)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "circo_submission.json")
    with open(save_path, 'w') as f:
        json.dump(predictions_dict, f)
        
    logging.info(f"Evaluation Done. Submission file saved to: {save_path}")
    print(f"\n[SUCCESS] JSON saved to {save_path}")



import torch
import logging
from tqdm import tqdm
from third_party.open_clip.clip import tokenize

def evaluate_genecis(model, img2text, args, dataloader):
    model.eval()
    img2text.eval()
    
    # 获取占位符 ID
    star_tokens = tokenize(["*"])
    placeholder_token_id = star_tokens[0][1].item()
    
    logging.info(f"Evaluating GeneCIS Task: {args.genecis_task}...")
    
    # 初始化计数器: R@1, R@2, R@3
    metrics = {k: 0 for k in [1, 2, 3]}
    total_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"GeneCIS-{args.genecis_task}"):
            ref_imgs = batch['ref_img'].cuda(args.gpu, non_blocking=True)
            captions = batch['caption']
            gallery_set = batch['gallery_set'].cuda(args.gpu, non_blocking=True)
            
            # [Batch, N_candidates, Channels, H, W]
            bsz, n_gallery, C, H, W = gallery_set.shape
            
            # --- 1. Query Feature ---
            ref_feats = model.encode_image(ref_imgs)
            soft_tokens = img2text(ref_feats)
            
            prompts = [f"a photo of * , {cap}" for cap in captions]
            texts = tokenize(prompts).cuda(args.gpu, non_blocking=True)
            
            query_feats = model.encode_text_img_retrieval(
                texts, soft_tokens, 
                split_ind=placeholder_token_id, 
                repeat=False
            )
            query_feats = query_feats / query_feats.norm(dim=-1, keepdim=True)
            
            # --- 2. Gallery Features ---
            # [修正点]：使用 bsz 和 n_gallery，而不是 B 和 N_gal
            gallery_flat = gallery_set.view(bsz * n_gallery, C, H, W)
            
            gallery_feats = model.encode_image(gallery_flat)
            gallery_feats = gallery_feats / gallery_feats.norm(dim=-1, keepdim=True)
            
            # Reshape back: [Batch, N_gallery, Dim]
            gallery_feats = gallery_feats.view(bsz, n_gallery, -1)
            
            # --- 3. Compute Similarity ---
            # [B, 1, D] x [B, D, N] -> [B, 1, N] -> [B, N]
            sims = torch.bmm(query_feats.unsqueeze(1), gallery_feats.transpose(1, 2)).squeeze(1)
            
            # --- 4. Calculate Metrics ---
            # 排序 (Descending)
            _, sorted_idxs = sims.sort(dim=-1, descending=True)
            
            # 正确答案永远在 index 0
            for k in [1, 2, 3]:
                # 截取前 K 个预测
                # 注意：GeneCIS 有些任务可能只有 2 个候选，取 Top3 不会报错，逻辑依然成立
                top_k = sorted_idxs[:, :k] 
                
                # 检查 0 是否命中
                hits = (top_k == 0).any(dim=1)
                metrics[k] += hits.sum().item()
            
            total_count += bsz
            
    # --- 5. 打印最终结果 ---
    print(f"\n====== GeneCIS Task: {args.genecis_task} Results ======")
    results = {}
    for k in [1, 2, 3]:
        acc = metrics[k] / total_count * 100
        results[k] = acc
        print(f"R@{k}: {acc:.2f}%")
        logging.info(f"GeneCIS {args.genecis_task} R@{k}: {acc:.2f}%")
        
    return results
