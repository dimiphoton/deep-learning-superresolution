# this code can be used to retrieve save model
from models.Generator import Generator, GeneratorV0
import torch

def load_gen(gen_args, filename, device="cuda:0"):
    """Returns a saved model

    Args:
        gen_args (dict): argument of the generator, must matched with the saved model
        filename (str): file where the model is saved
        device (str, optional): [description]. Defaults to "cuda:0".

    Returns:
        nn.Module: saved model
    """
    saved_gen = GeneratorV0(**gen_args)
    saved_gen.to(device)
    saved_gen.eval()
    checkpoint = torch.load(filename, map_location=torch.device(device))
    saved_gen.load_state_dict(checkpoint["model_state_dict"])
    return saved_gen


# Below are some saved models

def load_gen_02_05_2021(device="cuda:0"):
    # Upscale factor: 2 [-1, 1]
    # trainset=trainset, 96x96 hr images
    # batch_size=32,
    # epochs=200,
    # lr=0.0001,
    # gpu=True,
    # gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True}, 
    # dis_args={"nbr_channels": 64},
    # num_workers=4,
    # alpha=0.0002,
    # r1_penalty=0.005,
    saved_gen_02_05_2021 = GeneratorV0(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": True})
    saved_gen_02_05_2021.to(device)
    saved_gen_02_05_2021.eval()
    checkpoint = torch.load("models/save/gan-model-02-05-21-epochs200.pt", map_location=torch.device(device))
    saved_gen_02_05_2021.load_state_dict(checkpoint["model_state_dict"])
    return saved_gen_02_05_2021

def load_gen_03_05_2021(device="cuda:0"):
    # Upscale factor: 2 [-1, 1]
    # trainset=trainset, 96x96 hr images
    # batch_size=32,
    # epochs=300,
    # lr=0.0001,
    # gpu=True,
    # gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True}, 
    # dis_args={"nbr_channels": 64},
    # num_workers=4,
    # alpha=0.001,
    # r1_penalty=0.005,
    saved_gen_03_05_2021 = GeneratorV0(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": True})
    saved_gen_03_05_2021.to(device)
    saved_gen_03_05_2021.eval()
    checkpoint = torch.load("models/save/gan-model-03-05-21.pt", map_location=torch.device(device))
    saved_gen_03_05_2021.load_state_dict(checkpoint["model_state_dict"])
    return saved_gen_03_05_2021

def load_gen_04_05_2021(device="cuda:0"):
    # Upscale factor: 2 [-1, 1]
    # trainset=trainset, 96x96 hr images with COCO 40000+ images val2014
    # batch_size=32,
    # epochs=10,
    # lr=0.0001,
    # gpu=True,
    # gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True}, 
    # dis_args={"nbr_channels": 64},
    # num_workers=4,
    # alpha=0.001,
    # r1_penalty=0.005,
    saved_gen_04_05_2021 = GeneratorV0(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": True})
    saved_gen_04_05_2021.to(device)
    saved_gen_04_05_2021.eval()
    checkpoint = torch.load("models/save/gan-model-04-05-21.pt", map_location=torch.device(device))
    saved_gen_04_05_2021.load_state_dict(checkpoint["model_state_dict"])
    return saved_gen_04_05_2021


def load_gen_06_05_2021(device="cuda:0"):
    # Upscale factor: 4 [-1, 1]
    # trainset=trainset,
    # batch_size=32,
    # epochs=10,
    # lr=0.0001,
    # gpu=True,
    # gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": scaling_factor}, 
    # dis_args={"nbr_channels": 64},
    # num_workers=4,
    # alpha=0.001,
    # r1_penalty=0.01,
    # noisy_labels=True,
    try:
        saved_gen_05_05_2021 = Generator(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": False, "scaling_factor": 4})
        saved_gen_05_05_2021.to(device)
        saved_gen_05_05_2021.eval()
        checkpoint = torch.load("models/save/gan-model-06-05-21.pt", map_location=torch.device(device))
        saved_gen_05_05_2021.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print("The model 05_05_2021 does not exist")
        return
    return saved_gen_05_05_2021

def load_gen_05_05_2021(device="cuda:0"):
    # Upscale factor: 4 [-1, 1]
    # trainset=trainset,
    # batch_size=32,
    # epochs=10,
    # lr=0.0001,
    # gpu=True,
    # gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": scaling_factor}, 
    # dis_args={"nbr_channels": 64},
    # num_workers=4,
    # alpha=0.001,
    # r1_penalty=0.01,
    # noisy_labels=True,
    try:
        saved_gen_05_05_2021 = Generator(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": 4})
        saved_gen_05_05_2021.to(device)
        saved_gen_05_05_2021.eval()
        checkpoint = torch.load("models/save/gan-model-05-05-21-2.pt", map_location=torch.device(device))
        saved_gen_05_05_2021.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print("The model 05_05_2021 does not exist")
        return
    return saved_gen_05_05_2021

def load_gen_11_05_2021(device="cuda:0"):
    # model 11-05-21 [0, 1]
    # trainset=trainset, COCO
    #     testset=testset, DIv2K HR
    #     batch_size=16,
    #     epochs=8,
    #     lr=0.0001,
    #     gpu=True,
    #     gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": scaling_factor}, 
    #     dis_args={"nbr_channels": 64},
    #     num_workers=4,
    #     alpha=0.001,
    #     r1_penalty=0.01,
    #     labels="smooth",
    #     content_loss_type="SSIM",
    #     save_file="models/save/"+model_name+".pt")
    try:
        saved_gen_11_05_2021 = Generator(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": 4})
        saved_gen_11_05_2021.to(device)
        saved_gen_11_05_2021.eval()
        checkpoint = torch.load("models/save/gan-model-11-05-21.pt", map_location=torch.device(device))
        saved_gen_11_05_2021.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print("The model 05_05_2021 does not exist")
        return
    return saved_gen_11_05_2021


def load_gen_mse_ssim(device="cuda:0"):
    # gan-model-mse-ssim [0, 1]
    # # tratrainset=trainset,
    # testset=testset,
    # batch_size=32,
    # epochs=8,
    # lr=0.0001,
    # gpu=True,
    # gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": scaling_factor}, 
    # dis_args={"nbr_channels": 64},
    # num_workers=4,
    # alpha=0.001,
    # r1_penalty=0.01,
    # labels="smooth",
    # content_loss_type="MSE_SSIM",
    # save_file="models/save/"+model_name+".pt")
    try:
        saved_gen_11_05_2021 = Generator(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": 4})
        saved_gen_11_05_2021.to(device)
        saved_gen_11_05_2021.eval()
        checkpoint = torch.load("models/save/gan-model-mse-ssim-last.pt", map_location=torch.device(device))
        saved_gen_11_05_2021.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print("The model gan-model-mse-ssim does not exist")
        return
    return saved_gen_11_05_2021

def load_gen_lpips(device="cuda:0"):
    # gan-model-lpips-2 x4 [0, 1]
    # gen, dis, g_losses, d_losses, ssim_scores = train_gan(
    #     trainset=trainset,
    #     testset=testset,
    #     batch_size=32,
    #     epochs=4,
    #     lr=0.0001,
    #     gpu=True,
    #     gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": scaling_factor}, 
    #     dis_args={"nbr_channels": 64},
    #     num_workers=4,
    #     alpha=0.001,
    #     r1_penalty=0.01,
    #     labels="smooth",
    #     content_loss_type="MSE_SSIM",
    #     save_file="models/save/"+model_name+".pt")
    try:
        saved_gen_11_05_2021 = Generator(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": 4})
        saved_gen_11_05_2021.to(device)
        saved_gen_11_05_2021.eval()
        checkpoint = torch.load("models/save/gan-model-lpips-2.pt", map_location=torch.device(device))
        saved_gen_11_05_2021.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print("The model gan-model-mse-ssim does not exist")
        return
    return saved_gen_11_05_2021

def load_gen_mse_lpips(device="cuda:0"):
    # gan-model-lpips-2 [0, 1]
    # gen, dis, g_losses, d_losses, ssim_scores = train_gan(
    #     trainset=trainset,
    #     testset=testset,
    #     batch_size=32,
    #     epochs=4,
    #     lr=0.0001,
    #     gpu=True,
    #     gen_args={"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": scaling_factor}, 
    #     dis_args={"nbr_channels": 64},
    #     num_workers=4,
    #     alpha=0.001,
    #     r1_penalty=0.01,
    #     labels="smooth",
    #     content_loss_type="SSIM",
    #     save_file="models/save/"+model_name+".pt")
    try:
        saved_gen_11_05_2021 = Generator(**{"nbr_channels": 64, "nbr_blocks": 5, "normalize": True, "scaling_factor": 4})
        saved_gen_11_05_2021.to(device)
        saved_gen_11_05_2021.eval()
        checkpoint = torch.load("models/save/gan-model-mse-lpips.pt", map_location=torch.device(device))
        saved_gen_11_05_2021.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print("The model gan-model-mse-ssim does not exist")
        return
    return saved_gen_11_05_2021