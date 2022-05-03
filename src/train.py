import torch
from torch import nn
from models.Generator import Generator
from models.Discriminator import Discriminator
from time import time
from torch.optim.lr_scheduler import LambdaLR
from ignite.metrics import SSIM as igniteSSIM
from piqa import SSIM, LPIPS

#  SSIM Loss function
class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


# Function used to build and train a model
def train_model(model, optimizer, criterion, trainloader, testloader=None, epochs=10, test=False, plot=False, device="cpu"):
    epoch_train_losses = []
    epoch_test_losses = []

    # Main loop
    for i in range(epochs):
        tmp_loss = []
        for (x, y) in trainloader:
            outputs = model(x.to(device))
            loss = criterion(outputs, y.to(device))
            tmp_loss.append(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_losses.append(torch.tensor(tmp_loss).mean())
        print(f"Epoch {i+1}")

        if i % 10 == 9 and test:
            with torch.no_grad():
                tmp_loss = []
                for inputs, targets in testloader:
                    outputs = model(inputs.to(device))
                    loss = criterion(outputs, inputs)
                    tmp_loss.append(loss.detach())
                epoch_test_losses.append(torch.tensor(tmp_loss).mean())

    return epoch_train_losses, epoch_test_losses

def train_gan(  trainset: torch.utils.data.Dataset,
                testset: torch.utils.data.Dataset=None,
                lr: float = 0.0001,
                batch_size: int = 16,
                gpu: bool = True,
                save_file: str = None,
                use_amp: bool = True,
                log: bool = True,
                auto_tuner: bool = True,
                epochs: int = 4,
                gen_args: dict = None,
                dis_args: dict = None,
                num_workers: int=0, 
                alpha: float=0.001,
                r1_penalty: float=None, 
                labels="default", 
                content_loss_type="MSE"):
    """
    This function builds and train SRGANs. 

    Args:
        trainset (torch.utils.data.Dataset): training set
        testset (torch.utils.data.Dataset, optional): Test set for ssim scores at each epochs. Defaults to None.
        lr (float, optional): Base learning rate: it is used for the first half of the epochs,
                              then lr/10 is used for the rest. Defaults to 0.0001.
        batch_size (int, optional): batch size. Defaults to 16.
        gpu (bool, optional): Tells program use gpu ?. Defaults to True.
        save_file (str, optional): where the model will be saved. Defaults to None.
        use_amp (bool, optional): Tells program to use mixed-precision computation.
                                  Not compatible with SSIM loss functions. Defaults to True.
        log (bool, optional): Tells program to log some results. Defaults to True.
        auto_tuner (bool, optional): Tells to use auto-tuner functionality. Defaults to True.
        epochs (int, optional): Desired number of epochs. Defaults to 4.
        gen_args (dict, optional): Arguments of Generator, see models/Generator.py Defaults to None.
        dis_args (dict, optional): Arguments of Discriminator, see models/Discriminator.py. Defaults to None.
        num_workers (int, optional): Number of workers to use for data loaders. Defaults to 0.
        alpha (float, optional): Parameter balancing perceptual and adversaral losses.
                                 Loss = perceptual + alpha * adversarial. Defaults to 0.001.
        r1_penalty (float, optional): Discriminator gradient regularization parameter. Defaults to None.
        labels (str, optional): Should be in ["noisy", "smooth", "default].
                                "default": 0 for fake and 1 real.
                                "smooth": 0 for fake and 0.9 real.
                                "noisy": x~U[0, 0.2] for fake and x~U[0.8, 1] real.
                                Defaults to "default".
        content_loss_type (str, optional): Type of perceptual loss. Should be in ["MSE", "LPIPS", "SSIM", "MSE_LPIPS", "MSE_SSIM"].
                                        Defaults to "MSE".

    Returns:
        model, discriminator, epoch_train_losses, d_losses, ssim_scores: returns the model (generator), discriminator,
        iterations generator losses, iteration discriminator losses and ssim_scores only if a test set was given
    """

    plot_every_n_batches = 32

    # Initialization
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    if testset is not None:
            print("Warning: output range of images should be [0, 1] for testing")
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_ssim = igniteSSIM(data_range=1.0)
            ssim_scores = []
            best_ssim = 0.0


    torch.backends.cudnn.benchmark = auto_tuner
    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    model = Generator(**gen_args).to(device) if gen_args is not None else Generator().to(device)
    discriminator = Discriminator(**dis_args).to(device) if dis_args is not None else Discriminator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    if content_loss_type == "MSE_SSIM":
        print("Warning: output range of images should be [0, 1]")
        additional_loss = nn.MSELoss()
        criterion = SSIMLoss()
        criterion.to(device)
        use_amp = False
        ssim_loss = 0
        print("Cannot use mixed precision with SSIM")
    elif content_loss_type == "MSE_LPIPS":
        print("Warning: output range of images should be [0, 1]")
        additional_loss = nn.MSELoss()
        criterion = LPIPS()
        criterion.to(device)
        use_amp = False
        ssim_loss = 0
        print("Cannot use mixed precision with LPIPS")
    elif content_loss_type == "SSIM":
        print("Warning: output range of images should be [0, 1]")
        criterion = SSIMLoss() # SSIMLoss
        additional_loss = nn.MSELoss()
        criterion.to(device)
        use_amp = False
        ssim_loss = 0
        print("Cannot use mixed precision with SSIM")
    elif content_loss_type == "LPIPS":
        print("Warning: output range of images should be [0, 1]")
        criterion = LPIPS() # SSIMLoss
        additional_loss = nn.MSELoss()
        criterion.to(device)
        use_amp = False
        ssim_loss = 0
        print("Cannot use mixed precision with LPIPS")
    else:
        criterion = nn.MSELoss()

    d_criterion = nn.BCEWithLogitsLoss()

    scheduler = LambdaLR(optimizer, lambda epoch : 1 if epoch <= epochs // 2 else 0.1)
    d_scheduler = LambdaLR(d_optimizer, lambda epoch : 1 if epoch <= epochs // 2 else 0.1)

    # Training: main loop
    start = time()
    epoch_train_losses = []  # generator losses
    d_losses = []  # discriminator losses
    curr_batches = 0
    for i in range(epochs):
        epoch_start = time()
        tmp_loss = []
        d_tmp_loss = []

        # Training loop
        model.train()
        for (x, y) in trainloader:  # [batch_size x 3 x w x h]
            with torch.cuda.amp.autocast(enabled=use_amp):
                if labels == "noisy":
                    real_label = torch.rand((x.size(0), 1)).to(device) * 0.2 + 0.8
                    fake_label = torch.rand((x.size(0), 1)).to(device) * 0.2
                elif labels == "smooth":
                    real_label = torch.full((x.size(0), 1), 1, dtype=x.dtype).to(device) * 0.9
                    fake_label = torch.full((x.size(0), 1), 0, dtype=x.dtype).to(device)
                else:
                    real_label = torch.full((x.size(0), 1), 1, dtype=x.dtype).to(device)
                    fake_label = torch.full((x.size(0), 1), 0, dtype=x.dtype).to(device)

                # Update D
                discriminator.zero_grad(set_to_none=True)

                outputs = model(x.to(device))


                if r1_penalty is not None:
                    y.requires_grad = True

                real_scores = discriminator(y.to(device))
                d_loss_real = d_criterion(real_scores, real_label)
                d_loss_fake = d_criterion(discriminator(outputs.detach()), fake_label)
                d_loss = d_loss_real + d_loss_fake

                if r1_penalty is not None:
                    grad_real = torch.autograd.grad(outputs=real_scores.sum(), inputs=y, create_graph=True)[0]
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    d_loss += r1_penalty * grad_penalty

                d_tmp_loss.append(d_loss.detach())



                d_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0) 
                d_optimizer.step()

                # Update G

                # Perceptual loss
                model.zero_grad(set_to_none=True)
                if content_loss_type == "MSE_SSIM" :
                    additional_loss_value = additional_loss(outputs, y.to(device).detach())
                    ssim_loss = criterion(torch.clamp(outputs, 0.0, 1.0), y.to(device).detach())
                    content_loss = additional_loss_value + 0.2 * ssim_loss
                elif content_loss_type == "SSIM":
                    ssim_loss = criterion((torch.clamp(outputs, 0.00, 1.0)), y.to(device).detach())
                    if torch.isnan(ssim_loss):
                        print("content loss is nan")
                        ssim_loss = additional_loss(outputs, y.to(device).detach())
                    content_loss = 0.1 * ssim_loss
                elif content_loss_type == "LPIPS":
                    ssim_loss = criterion((torch.clamp(outputs, 0.00, 1.0)), y.to(device).detach())
                    if torch.isnan(ssim_loss):
                        print("content loss is nan")
                        ssim_loss = additional_loss(outputs, y.to(device).detach())
                    content_loss = 0.1 * ssim_loss
                elif content_loss_type == "MSE_LPIPS":
                    additional_loss_value = additional_loss(outputs, y.to(device).detach())
                    lpips_loss = criterion((torch.clamp(outputs, 0.00, 1.0)), y.to(device).detach())
                    if torch.isnan(lpips_loss):
                        print("content loss is nan")
                        lpips_loss = additional_loss(outputs, y.to(device).detach())
                    content_loss = additional_loss_value + 0.2 * lpips_loss
                else:
                    content_loss = criterion(outputs, y.to(device).detach())
                
                if labels != "default":
                    real_label = torch.full((x.size(0), 1), 1, dtype=x.dtype).to(device)
                gan_loss = d_criterion(discriminator(outputs), real_label)
                g_loss = content_loss + alpha * gan_loss

                tmp_loss.append(g_loss.detach())

                g_loss.backward()
                optimizer.step()

                curr_batches += 1
                if curr_batches % plot_every_n_batches == plot_every_n_batches-1:
                    # append mean loss per sample
                    d_losses.append(torch.tensor(d_tmp_loss).mean())
                    epoch_train_losses.append(torch.tensor(tmp_loss).mean())
                    d_tmp_loss = []
                    tmp_loss = []
                    if log:
                        try:
                            print(f"Iteration {curr_batches}, total time: {(time() - start):.2f}s,  loss: {epoch_train_losses[-1]:.8f}")
                        except IndexError:
                            print(f"Iteration {curr_batches}, total time: {(time() - start):.2f}s")
                
                

        # Logging and saving
        torch.cuda.empty_cache()
        if log:
            try:
                print(f"Epoch {i+1} in {(time() - epoch_start):.2f}s, total time: {(time() - start):.2f}s, lr: {scheduler.get_last_lr()},  loss: {epoch_train_losses[-1]:.8f}, remaing time: {(epochs - i - 1) * (time() - epoch_start):.2f}s")
            except IndexError:
                print(f"Epoch {i+1} in {(time() - epoch_start):.2f}s, total time: {(time() - start):.2f}s, lr: {scheduler.get_last_lr()}, remaing time: {(epochs - i - 1) * (time() - epoch_start):.2f}s")
        if save_file is not None:
            try:
                torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'd_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': epoch_train_losses[-1],
            }, save_file)
            except IndexError:
                # no saving
                pass
        
        scheduler.step()
        d_scheduler.step()

        # Evaluation : image quality with ssim and save best model
        if testset is not None and i % 1 == 0:
            model.eval()
            with torch.no_grad():
                test_ssim.reset()
                for (x, y) in testloader:  # [batch_size x 3 x w x h]
                    outputs = model(x.to(device))
                    test_ssim.update((outputs, y.to(device)))
                ssim_score =  test_ssim.compute()
                ssim_scores.append(ssim_score.detach())
                print(f'Epoch {i+1}, ssim score: {ssim_score}')
                test_ssim.reset()
                if ssim_score > best_ssim and save_file is not None:
                    best_ssim = ssim_score
                    try:
                        torch.save({
                        'epoch': i+1,
                        'model_state_dict': model.state_dict(),
                        'd_state_dict': discriminator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'd_optimizer_state_dict': d_optimizer.state_dict(),
                        'g_loss': epoch_train_losses[-1],
                    }, save_file + 'best_ssim')
                    except IndexError:
                        # no saving
                        pass
        
        stats = torch.cuda.memory_stats()
        print(f"peak: {stats['allocated_bytes.all.peak']}  curr: {stats['allocated_bytes.all.current']}")



    end = time()
    print(f"Training took {end - start:.2f} seconds for {epochs} epochs, or {(end - start)/epochs:.2f} seconds per epochs")
    try:
        print(f"Final loss {epoch_train_losses[-1]:.8f}")
    except IndexError:
        pass
    
    if testset is not None:
        return model, discriminator, epoch_train_losses, d_losses, ssim_scores
    return model, discriminator, epoch_train_losses, d_losses, None
