import torch
import torchvision
from torchvision import transforms
from torch.nn.functional import cosine_similarity

import torch
import torchvision
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import json


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()
def read_images(path):
    return torchvision.io.read_image(path)

def image_preprocessing(image, height, width):
    """
        args
        image -> input tensor ( ..., H, W )
    """
    resize = transforms.Resize( ( height, width), antialias=True )
    resized_image = resize(image)
    print( resized_image.shape)
    return resized_image.float()


def compute_embedding( image ):
    image = image.unsqueeze(0)
    print( model(image).shape )
    return model(image)
        
"""
    { prodouctId : image_embd }
"""
def compute_similar_images( embd, num_images, product ):
    all_embds = torch.stack(list(product.values()), dim = 0)
    all_products =  list(product.keys())


    similarity_scores = cosine_similarity( embd, all_embds, dim = 1 )

    topk_indices = torch.topk( similarity_scores, num_images, dim = -1 ).indices

    print( topk_indices)
    topk_products = [  all_products[i] for i in topk_indices ]

    return topk_products

def read_embeddings():
    f = open("products.json", "r")
    data = f.read()

    data = json.loads(data)

    for product in data.items(): 
        
        data[product[0]] = torch.tensor(product[1])
        # print( data[product[0]].dtype )
    return data


image = read_images('./images/95534.jpg')
resized_image = image_preprocessing(image, 800, 800)
product = read_embeddings()
print(compute_similar_images( compute_embedding(resized_image), 5, product ))

# compute_embedding( resized_image)
