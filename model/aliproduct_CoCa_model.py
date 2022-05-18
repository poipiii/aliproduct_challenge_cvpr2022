import torch

# import vision transformer

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

vit = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

# do your vision transformer training

vit = Extractor(vit, return_embeddings_only = True)

# extractor will enable it so the vision transformer returns its embeddings

# import CoCa and instantiate it

from coca_pytorch.coca_pytorch import CoCa

coca = CoCa(
    dim = 512,                     # model dimension
    img_encoder = vit,             # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
    image_dim = 1024,              # image embedding dimension, if not the same as model dimensions
    num_tokens = 20000,            # number of text tokens
    unimodal_depth = 6,            # depth of the unimodal transformer
    multimodal_depth = 6,          # depth of the multimodal transformer
    dim_head = 64,                 # dimension per attention head
    heads = 8,                     # number of attention heads
    caption_loss_weight = 1.,      # weight on the autoregressive caption loss
    contrastive_loss_weight = 1.,  # weight on the contrastive loss between image and text CLS embeddings
).cuda()

# mock text and images

text = torch.randint(0, 20000, (4, 512)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# train by giving CoCa your text and images with `return_loss = True`

loss = coca(
    text = text,
    images = images,
    return_loss = True  # set this to True to get the full caption + contrastive loss
)

loss.backward()

# do the above for as much text and images...
# then you can get the caption logits as so

logits = coca(
    text = text,
    images = images
) # (4, 512, 20000)

# and the CLIP-like text and image embeddings as

text_embeds, image_embeds = coca(
    text = text,
    images = images,
    return_embeddings = True
) # (4, 512), (4, 512)

print(images.size())