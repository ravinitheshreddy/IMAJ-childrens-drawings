# Code courtesy of Ludovica (https://github.com/ludovicaschaerf/Cini_TDA)

# Create a super model that can use a pre trained model

from typing import List

import torch
from torch import nn
from torchvision import models


class DrawingsNet(torch.nn.Module):
    def __init__(self, model_name: str, device: torch.device, pooling: str = "avg"):
        """Initializes a pre trained CNN model excluding the Fully Connected Layers.

        Parameters
        ----------
        model_name : str
            String name of pretrained model. Options available: resnet50, resnet100, resnet152, densenet161, resnext-101, regnet_y_32gf, vit_b_16, convnext_tiny, efficientnet0, efficientnet7
        device : torch.device
            Device to run the forward pass.
        pooling : str
            String to indicate pooling layer to use before the fully connected layer. Options available: avg or max.
        """

        super(DrawingsNet, self).__init__()

        # select the model based on user input
        if model_name == "resnet18":
            selected_model = models.resnet18(pretrained=True)
        elif model_name == "resnet50":
            selected_model = models.resnet50(pretrained=True)
        elif model_name == "resnet101":
            selected_model = models.resnet101(pretrained=True)
        elif model_name == "resnet152":
            selected_model = models.resnet152(pretrained=True)
        elif model_name == "densenet161":
            selected_model = models.densenet161(pretrained=True)
        elif model_name == "resnext-101":
            selected_model = models.resnext101_32x8d(pretrained=True)
        elif model_name == "regnet_y_32gf":
            selected_model = models.regnet_y_32gf(pretrained=True)
        elif model_name == "vit_b_16":
            selected_model = models.vit_b_16(pretrained=True)
        elif model_name == "convnext_tiny":
            selected_model = models.convnext_tiny(pretrained=True)
        elif model_name == "efficientnet0":
            selected_model = models.efficientnet_b0(pretrained=True)
        elif model_name == "efficientnet7":
            selected_model = models.efficientnet_b7(pretrained=True)
        else:
            raise Exception(
                "model_name should be any of resnet18, resnet50, resnet100, resnet152, densenet161, resnext-101, regnet_y_32gf, vit_b_16, convnext_tiny, efficientnet0, efficientnet7"
            )

        # set the device to run the forward pass
        self.device = device

        # The pooling is applied depending on the selection
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1)).to(self.device)

        # The layers before pooling are selected and moved to the device
        self.non_pooled = torch.nn.Sequential(
            *(list(selected_model.children())[:-2])
        ).to(self.device)

        # The fully connected layer of the model is set to identity, so that the input to it is returned unaltered.
        selected_model.fc = nn.Identity()

        # The fully connected layers and the model are moved to the device
        self.fc = selected_model.fc.to(self.device)

        self.model = selected_model.to(self.device)

    def forward(
        self,
        base_image: torch.Tensor,
        similar_image: torch.Tensor,
        dissimilar_image: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Returns the normalised embeddings of a triplet of images.

        Parameters
        ----------
        base_image : torch.Tensor
            The image to be compared
        similar_image : torch.Tensor
            The image that is supposed to be similar to the base image.
        dissimilar_image : torch.Tensor
            The image that is not to be similar to the base image.
        
        
        Returns
        -------
        base_image_norm : torch.Tensor
            The feature vector of the base image normalized to 1.
        similar_image_norm : torch.Tensor
            The feature vector of the similar image normalized to 1.
        dissimilar_image_norm : torch.Tensor
            The feature vector of the dissimilar image normalized to 1.
        """

        base_image_emb = self.model(base_image)
        similar_image_emb = self.model(similar_image)
        dissimilar_image_emb = self.model(dissimilar_image)

        base_image_norm = torch.div(
            base_image_emb, torch.linalg.vector_norm(base_image_emb)
        )
        similar_image_norm = torch.div(
            similar_image_emb, torch.linalg.vector_norm(similar_image_emb)
        )
        dissimilar_image_norm = torch.div(
            dissimilar_image_emb, torch.linalg.vector_norm(dissimilar_image_emb)
        )

        return base_image_norm, similar_image_norm, dissimilar_image_norm

    def non_pooled_forward(
        self,
        base_image: torch.Tensor,
        similar_image: torch.Tensor,
        dissimilar_image: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Returns the non pooled and non normalized feature vectors and normalised embeddings of a triplet of images.

        Parameters
        ----------
        base_image : torch.Tensor
            The image to be compared
        similar_image : torch.Tensor
            The image that is supposed to be similar to the base image.
        dissimilar_image : torch.Tensor
            The image that is not to be similar to the base image.
        
        
        Returns
        -------
        base_image_np : torch.Tensor
            The feature vector of the base image without pooling.
        similar_image_np : torch.Tensor
            The feature vector of the similar image without pooling.
        dissimilar_image_np : torch.Tensor
            The feature vector of the dissimilar image without pooling.
        base_image_norm : torch.Tensor
            The feature vector of the base image normalized to 1.
        similar_image_norm : torch.Tensor
            The feature vector of the similar image normalized to 1.
        dissimilar_image_norm : torch.Tensor
            The feature vector of the dissimilar image normalized to 1.
        """

        base_image_np = self.non_pooled(base_image)
        similar_image_np = self.non_pooled(similar_image)
        dissimilar_image_np = self.non_pooled(dissimilar_image)

        base_image_p = self.pool(base_image_np)
        similar_image_p = self.pool(similar_image_np)
        dissimilar_image_p = self.pool(dissimilar_image_np)

        base_image_emb = self.fc(base_image_p)
        similar_image_emb = self.fc(similar_image_p)
        dissimilar_image_emb = self.fc(dissimilar_image_p)

        base_image_norm = torch.div(
            base_image_emb, torch.linalg.vector_norm(base_image_emb)
        )
        similar_image_norm = torch.div(
            similar_image_emb, torch.linalg.vector_norm(similar_image_emb)
        )
        dissimilar_image_norm = torch.div(
            dissimilar_image_emb, torch.linalg.vector_norm(dissimilar_image_emb)
        )

        return (
            base_image_np,
            similar_image_np,
            dissimilar_image_np,
            base_image_norm,
            similar_image_norm,
            dissimilar_image_norm,
        )

    def size(self, image_sample: torch.Tensor) -> int:
        """Returns the size of a image.

        Parameters
        ----------
        image_sample : torch.Tensor
            The image to get the size
        
        
        Returns
        -------
        num_features : int
            The size the image tensor.
        """
        size = image_sample.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, image_sample: torch.Tensor):
        """Returns the normalised embeddings of a single image.

        Parameters
        ----------
        image_sample : torch.Tensor
            The image to get the embedding
        
        
        Returns
        -------
        image_sample_norm : torch.Tensor
            The feature vector of the image normalized to 1.
            
        """

        self.model.eval()
        image_sample_emb = self.model(image_sample)
        image_sample_norm = torch.div(
            image_sample_emb,
            torch.linalg.vector_norm(image_sample_emb, dim=1).unsqueeze(-1),
        )
        return image_sample_norm

    def predict_non_pooled(self, image_sample):
        """Returns the embeddings of a single image before pooling and normalization.

        Parameters
        ----------
        image_sample : torch.Tensor
            The image to get the embedding
        
        
        Returns
        -------
        image_sample_np : torch.Tensor
            The feature vector of the image before pooling and normalization.
            
        """
        image_sample_np = self.non_pooled(image_sample)
        return image_sample_np
