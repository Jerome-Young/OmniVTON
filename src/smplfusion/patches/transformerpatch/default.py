import torch 
from ... import share


def forward(self, x, context=None, in_mask=None, out_mask=None, outpainting=False):
    x = x + self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, in_mask=in_mask,
                       out_mask=out_mask, outpainting=outpainting)  # Self Attn.
    x = x + self.attn2(self.norm2(x), context=context, in_mask=in_mask, out_mask=out_mask) # Cross Attn.
    x = x + self.ff(self.norm3(x))
    return x
