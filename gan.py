from pytorch_lightning import LightningModule
from src.dataloader.memory_buffer import Memory
from src.network.models import Generator, Discriminator
import torch
import torch.nn.functional as F


class GAN(LightningModule):
    def __init__(
        self,
        channels=3,
        width=32,
        height=32,
        latent_dim: int = 2,
        lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 16,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.memo = Memory(2000)
        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(self.hparams.latent_dim + 1).to("cuda")
        self.discriminator = Discriminator(4).to("cuda")

        # self.sinkhorn = SinkhornDistance (eps = 0.1, max_iter = 100, reduction = "mean").cuda()

        self.validation_z = torch.randn([10, 2, 4, 4]) / 0.50 - 0.25

        classes = (torch.tensor(list(range(10))).view(-1, 1, 1, 1).float() - 5) / 10
        classes = classes.repeat(1, 1, 4, 4)

        self.validation_z = torch.cat([self.validation_z, classes], axis=1)

        self.example_input_array = torch.zeros([10, 2, 4, 4])
        self.example_input_array = torch.cat(
            [self.example_input_array, classes], axis=1
        )
        self.loss_fn_alex = lpips.LPIPS(net="alex")

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, real_dst, fake_dst):
        # dist, P, C = self.sinkhorn (real_dst, fake_dst)
        # return dist
        return F.mse_loss(fake_dst, torch.zeros_like(fake_dst)) + F.mse_loss(
            real_dst, torch.ones_like(real_dst)
        )

    def class_loss(self, img_fake, img_real):
        img_fake = F.upsample(img_fake, (64, 64))
        img_real = F.upsample(img_real, (64, 64))
        loss = self.loss_fn_alex(img_fake, img_real, retPerLayer=False)
        # dist, P, C = self.sinkhorn (y_hat.flatten(1), y.flatten(1))
        return loss.mean()

    # def gradient_penalty(self, netD, real_data, fake_data, lambda_val=10):

    #     #Interpolate Between Real and Fake data
    #     shape = [real_data.size(0)] + [1] * (real_data.dim() - 1)
    #     alpha = torch.rand(shape).cuda()
    #     z = real_data + alpha * (fake_data - real_data)

    #     # Compute Gradient Penalty
    #     z = torch.autograd.Variable(z, requires_grad=True).cuda()
    #     disc_z = netD(z)

    #     gradients = torch.autograd.grad(outputs=disc_z, inputs=z,
    #                             grad_outputs=torch.ones(disc_z.size()).cuda(),
    #                             create_graph=True)[0].view(z.size(0), -1)

    #     gradient_penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean() * lambda_val
    #     return gradient_penalty

    def gradient_penalty(self, netD, real_data, generated_data):
        batch_size = min(real_data.size()[0], generated_data.size()[0])

        # Calculate interpolation
        alpha = torch.rand(generated_data.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(generated_data)
        alpha = alpha.cuda()
        interpolated = (
            alpha[:batch_size] * real_data[:batch_size].data
            + (1 - alpha[:batch_size]) * generated_data.data[:batch_size]
        )
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
        interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = netD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda()
            if True
            else torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()

    def training_step(self, batch, batch_idx):
        imgs, lbls = batch
        imgs = imgs.to("cuda")

        g_opt, d_opt = self.optimizers()

        # sample noise

        z = torch.randn([imgs.shape[0], 2, 4, 4]).to("cuda") / 0.50 - 0.25
        classes = (lbls.view(-1, 1, 1, 1).float().to("cuda") - 5) / 10
        classes_ = classes.repeat(1, 1, 4, 4)
        z = torch.cat([z, classes_], axis=1)
        classes_img_shape = classes.repeat(1, 1, *imgs.shape[-2:])

        disc_gen = self(z).detach()
        fake_img = torch.cat([disc_gen, classes_img_shape], axis=1)
        real_img = torch.cat([imgs, classes_img_shape], axis=1)

        # discriminator
        real_dst = self.discriminator(real_img)
        fake_dst = self.discriminator(fake_img)
        self.d_loss = self.adversarial_loss(real_dst, fake_dst)
        self.gp = self.gradient_penalty(self.discriminator, real_img, fake_img)
        d_los = self.d_loss + self.gp
        # discriminator loss is the average of these
        # d_loss = (real_loss + fake_loss) / 2
        d_opt.zero_grad()
        self.manual_backward(self.d_loss, retain_graph=True)
        # d_opt.step()
        [
            self.memo.add(loss.detach().mean().cpu(), im.detach().unsqueeze(0).cpu())
            for im, loss in zip(fake_img, fake_dst)
        ]
        samples, ids, losses = self.memo.sample(64)
        fake_img = torch.stack(samples).to("cuda").squeeze(1)

        self.d_loss = self.adversarial_loss(real_dst, fake_dst)
        self.gp = self.gradient_penalty(self.discriminator, real_img, fake_img)
        d_los = self.d_loss + self.gp
        # discriminator loss is the average of these
        # d_loss = (real_loss + fake_loss) / 2
        d_opt.zero_grad()
        self.manual_backward(self.d_loss)
        d_opt.step()

        [
            self.memo.update(id, loss.detach().mean().cpu())
            for id, loss in zip(ids, fake_dst)
        ]

        # generator
        self.generated_imgs = self(z)
        valid = torch.ones(z.shape[0], 1)
        valid = valid.type_as(z)
        generated = self(z)
        fake_img = torch.cat([generated, classes_img_shape], axis=1)

        real_dst = self.discriminator(real_img)
        fake_dst = self.discriminator(fake_img)

        g_loss = self.adversarial_loss(real_dst, fake_dst) * -1
        perception_loss = self.class_loss(generated, imgs)

        gc_loss = g_loss + perception_loss
        g_opt.zero_grad()
        self.manual_backward(gc_loss / 2)
        g_opt.step()

        # self.log("train_loss", {"d_loss": d_loss}, on_step=False, on_epoch=True, logger=True)
        tqdm_dict = {
            "g_loss": g_loss.detach(),
            "d_loss": self.d_loss.detach(),
            "perception_loss": perception_loss.detach(),
            "gp loss": self.gp.detach(),
            "gc_loss": gc_loss.detach(),
        }
        self.log(
            "train_loss",
            {
                "g_loss": g_loss,
                "d_loss": self.d_loss,
                "perception_loss": perception_loss,
                "gp loss": self.gp,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        output = OrderedDict(
            {
                "d_loss": self.d_loss.detach(),
                "g_loss": gc_loss.detach(),
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
        )
        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)

        return opt_g, opt_d

    def on_epoch_end(self):
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        class_names = self.validation_z[
            :,
        ]

        self.validation_z = torch.randn([10, 2, 4, 4]) / 0.50 - 0.25

        classes = (torch.tensor(list(range(10))).view(-1, 1, 1, 1).float() - 5) / 10
        classes = classes.repeat(1, 1, 4, 4)

        self.validation_z = torch.cat([self.validation_z, classes], axis=1)
        z = self.validation_z.float().cuda()

        # log sampled images
        sample_imgs = self(z)
        sample_imgs = F.upsample(sample_imgs, size=(128, 128))
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
