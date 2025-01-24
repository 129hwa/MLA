import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

class DomainLabeledDataset(Dataset):
    """
    기존 ImageFolder 등을 받아
    (image, domain_label, class_label)을 반환하도록 래핑
    """
    def __init__(self, base_dataset, domain_label):
        self.base_dataset = base_dataset
        self.domain_label = domain_label

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, class_label = self.base_dataset[idx]
        return image, self.domain_label, class_label

data_path = './data/PACS/'
BATCH_SIZE = 128
all_domains = ["photo", "art_painting", "cartoon", "sketch"]

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 각 도메인별 데이터셋 생성
all_datasets = []
for domain_idx, domain_name in enumerate(all_domains):
    domain_full_path = os.path.join(data_path, domain_name)
    base_dataset = ImageFolder(domain_full_path, transform=data_transform)
    wrapped_dataset = DomainLabeledDataset(base_dataset, domain_idx)
    all_datasets.append(wrapped_dataset)

# 첫 번째 도메인(test_domain_idx=0)을 테스트로, 나머지를 학습용
test_domain_idx = 0
test_dataset = all_datasets[test_domain_idx]
train_datasets = [ds for i, ds in enumerate(all_datasets) if i != test_domain_idx]
train_dataset = ConcatDataset(train_datasets)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ---------------------------
# (A) Gradient Reversal
# ---------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

# ---------------------------
# (B) JVAE Encoder
# ---------------------------
import torch
import numpy as np
# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
# from .types_ import *


LR = 0.0001
train_batch_size = 128
val_batch_size = 128
patch_size = 64
num_workers = 2



class JointVAE(nn.Module):
    num_iter = 1

    def __init__(self,
                 in_channels = 3,
                 latent_dim = 512,
                 categorical_dim = 7,
                 **kwargs) -> None:
        super(JointVAE, self).__init__()

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

        self.temp = 0.9
        self.min_temp = 0.7
        self.anneal_rate = 0.00003
        self.anneal_interval = 200
        self.alpha = 50.

        self.cont_min = 0.0
        self.cont_max = 10.0

        self.disc_min = 0.0
        self.disc_max = 10.0

        self.cont_gamma = 20.
        self.disc_gamma = 20.

        self.cont_iter = 25000
        self.disc_iter = 25000


        self.encoder = nn.Sequential(
            # 첫 번째
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU()
            ),
            # 두 번째
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU()
            ),
            # 세 번째
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU()
            ),
            # 네 번째
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU()
            ),
            # 다섯 번째
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU()
            )
        )
        self.fc_mu = nn.Linear(512*7*7, self.latent_dim)
        self.fc_var = nn.Linear(512*7*7, self.latent_dim)
        self.fc_z = nn.Linear(512*7*7, self.categorical_dim)

        # Build Decoder

        self.decoder_input = nn.Linear(self.latent_dim + self.categorical_dim,
                                       512*7*7)

        self.decoder = nn.Sequential(
            # 첫 번째
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, 
                                 padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU()
            ),
            # 두 번째
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                 padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU()
            ),
            # 세 번째
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                 padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU()
            ),
            # 네 번째
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                 padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Tanh()
            )
        )

        # self.sampling_dist = torch.distributions.OneHotCategorical(1. / categorical_dim * torch.ones((self.categorical_dim, 1)))

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) Latent code [B x D x Q]
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = self.fc_z(x)
        # z = z.view(-1, self.categorical_dim)
        return mu, log_var, z

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x Q]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 7, 7)
        result = self.decoder(result)
        return result

    def reparameterize(self,mu,log_var,q):
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param mu: (Tensor) mean of the latent Gaussian  [B x D]
        :param log_var: (Tensor) Log variance of the latent Gaussian [B x D]
        :param q: (Tensor) Categorical latent Codes [B x Q]
        :return: (Tensor) [B x (D + Q)]
        """

        std = torch.exp(0.5 * log_var)
        e = torch.randn_like(std)
        z = e * std + mu

        # Gumbel로부터 샘플링
        u = torch.rand_like(q)
        g = - torch.log(- torch.log(u + 1e-7) + 1e-7)

        # Gumbel-Softmax 샘플
        s = F.softmax((q + g) / 0.5, dim=-1)
        return torch.cat([z, s], dim=1)


    def forward(self, input):
        mu, log_var, q = self.encode(input)
        z = self.reparameterize(mu, log_var, q)
        return  self.decode(z), input, q, mu, log_var

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        q = args[2]
        mu = args[3]
        log_var = args[4]

        q_p = F.softmax(q, dim=-1)


        kld_weight = kwargs['M_N']
        batch_idx = kwargs['batch_idx']

        # Anneal the temperature at regular intervals
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
                                   self.min_temp)

        recons_loss =  F.mse_loss(recons, input) + 0.5 * F.l1_loss(recons, input) #recons_loss =F.mse_loss(recons, input, reduction='mean')

        disc_curr = (self.disc_max - self.disc_min) * \
                    self.num_iter/ float(self.disc_iter) + self.disc_min
        disc_curr = min(disc_curr, np.log(self.categorical_dim))


        eps = 1e-7

        h1 = q_p * torch.log(q_p + eps) # 로짓 엔트로피
        h2 = q_p * np.log(1. / self.categorical_dim + eps) # 크로스 엔트로피
        kld_disc_loss = torch.mean(torch.sum(h1 - h2, dim =1), dim=0)

        # 연속형 손실 계산
        cont_curr = (self.cont_max - self.cont_min) * \
                    self.num_iter/ float(self.cont_iter) + self.cont_min
        cont_curr = min(cont_curr, self.cont_max)

        kld_cont_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),
                                                    dim=1),
                                   dim=0)
        capacity_loss = self.disc_gamma * torch.abs(disc_curr - kld_disc_loss) + \
                        self.cont_gamma * torch.abs(cont_curr - kld_cont_loss)
        # kld_weight = 1.2
        loss = self.alpha * recons_loss + kld_weight * capacity_loss

        if self.training:
            self.num_iter += 1
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Capacity_Loss':capacity_loss}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        # [S x D]
        z = torch.randn(num_samples,
                        self.latent_dim)

        M = num_samples
        np_y = np.zeros((M, self.categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(self.categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M , self.categorical_dim])
        q = torch.from_numpy(np_y)

        # z = self.sampling_dist.sample((num_samples * self.latent_dim, ))
        z = torch.cat([z, q], dim = 1).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

# ---------------------------
# (D) Domain Classifier
# ---------------------------
class DomainClassifier(nn.Module):
    """
    z_inv만 입력 -> 도메인 예측 (P/A/C/S)
    Gradient Reversal로 Encoder는 도메인 분류가 어렵게 학습
    """
    def __init__(self, z_dim_inv=512+7, num_domains=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim_inv, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains)
        )
    
    def forward(self, z_inv, alpha=0.90):
        z_rev = grad_reverse(z_inv, alpha)
        return self.net(z_rev)

def save_samples(model, device, output_dir='samples', num_samples=16):
    """
    무작위 latent에서 이미지를 생성하고 저장

    :param model: 학습된 JointVAE
    :param output_dir: 결과 이미지 저장 폴더
    :param num_samples: 생성할 이미지의 개수
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, current_device=device)
    
    # grid 형태로 저장
    file_path = os.path.join(output_dir, "random_samples.png")
    torchvision.utils.save_image(samples, file_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"[save_samples] Saved random samples to '{file_path}'")

import os
import torchvision

def save_reconstructions(model, loader, device, output_dir='recons', num_images=5):
    """
    DataLoader에서 배치를 하나 추출
    모델 forward를 통해 재구성된 이미지
    원본 이미지와 재구성 이미지를 각각 파일로 저장

    :param loader: 테스트 DataLoader (image, domain_label, class_label) 반환
    :param num_images: 저장할 샘플 이미지 개수 (배치 중 앞 부분만)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    
    # DataLoader에서 배치 하나를 가져옴
    # (images, domain_labels, class_labels) 형태
    images, domain_lbl, class_lbl = next(iter(loader))
    images = images.to(device)
    
    with torch.no_grad():
        # 모델의 forward 결과 (recons, input, q, mu, log_var) 
        # forward(x) -> (재구성이미지, 원본, q, mu, logvar)
        recons, _, _, _, _ = model(images)
    
    # 원하는 개수만큼 저장
    for i in range(num_images):
        # 원본 이미지 저장
        torchvision.utils.save_image(
            images[i], 
            os.path.join(output_dir, f"original_{i}.png")
        )
        # 재구성 이미지 저장
        torchvision.utils.save_image(
            recons[i],
            os.path.join(output_dir, f"reconstructed_{i}.png")
        )
        
    print(f"[save_reconstructions] Saved {num_images} pairs of original/reconstructed images to '{output_dir}'")


def domain_classification_loss(logits, domain_labels):
    return F.cross_entropy(logits, domain_labels)

# ---------------------------
# (F) 학습 루프 (JVAE)
# ---------------------------
def train_jvae_only(model, train_loader, optimizer, device='cuda'):
    """
    Domain Classifier 없이 JointVAE만 단독 학습
    """
    model.train()
    
    total_loss = 0.0
    for batch_idx, (x, d_lbl, c_lbl) in enumerate(train_loader):
        x = x.to(device)
        
        # Forward VAE
        recon_x, _, q, mu, log_var = model(x)
        
        # VAE Loss
        # M_N=0.00025, batch_idx=batch_idx
        loss_dict = model.loss_function(
            recon_x, x, q, mu, log_var,
            M_N=0.00025, batch_idx=batch_idx
        )
        vae_loss = loss_dict['loss']
        
        optimizer.zero_grad()
        vae_loss.backward()
        optimizer.step()
        
        total_loss += vae_loss.item()
    
    return total_loss / len(train_loader)
# ---------------------------
# (F) 학습 루프 (JVAE + Domain)
# ---------------------------

def train_jvae_domain(model, domain_cls,
                      train_loader,
                      optimizer, 
                      device='cuda',
                      domain_loss_weight=1.0,
                      grad_rev_alpha=0.90):
    """
    JointVAE + Domain Classifier(gradient reversal) 학습
    - domain_loss_weight: 도메인 분류 손실 가중치
    - grad_rev_alpha: gradient reversal 계수
    """
    model.train()
    domain_cls.train()
    
    total_vae_loss = 0.0
    total_dom_loss = 0.0
    
    for batch_idx, (x, d_lbl, c_lbl) in enumerate(train_loader):
        x = x.to(device)
        d_lbl = d_lbl.to(device)
        
        # Forward VAE
        recon_x, _, q, mu, log_var = model(x)
        
        # VAE Loss
        loss_dict = model.loss_function(
            recon_x, x, q, mu, log_var, 
            M_N=0.00025, 
            batch_idx=batch_idx
        )
        vae_loss = loss_dict['loss']
        
        # Domain Classification
        # z_inv = [mu, q]인 지점은 기존 코드와 동일
        z = torch.cat([mu, q], dim=1)
        dom_logits = domain_cls(z, alpha=grad_rev_alpha)
        dom_loss = F.cross_entropy(dom_logits, d_lbl)
        
        # 최종 Loss = VAE + (domain_loss_weight) * 도메인 손실
        total_loss = vae_loss + domain_loss_weight * dom_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_vae_loss += vae_loss.item()
        total_dom_loss += dom_loss.item()
    
    avg_vae = total_vae_loss / len(train_loader)
    avg_dom = total_dom_loss / len(train_loader)
    return avg_vae, avg_dom

class ClassLabelClassifier(nn.Module):
    """
    JVAE의 latent (z_inv, z_sp) -> 클래스 라벨 예측
    """
    def __init__(self, z_dim_inv=32, z_dim_sp=32, num_classes=7):
        super().__init__()
        in_dim = z_dim_inv + z_dim_sp
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, z_inv, z_sp):
        z = torch.cat([z_inv, z_sp], dim=1)  # [B, z_dim_inv+z_dim_sp]
        logits = self.net(z)
        return logits

def train_class_classifier(model, classifier, train_loader, 
                           opt_cls,
                           device='cuda',
                           freeze_model=True,
                           num_epochs=50):
    """
    VAE 이미 학습된 상태에서
    latent -> class label 분류기만 학습
    """
    if freeze_model:
        model.eval()  
        for param in model.parameters():
            param.requires_grad = False
    
    classifier.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for (x, d_lbl, c_lbl) in train_loader:
            x = x.to(device)
            d_lbl = d_lbl.to(device)
            c_lbl = c_lbl.to(device)
            
            with torch.set_grad_enabled(not freeze_model):
                # z_inv, _, _, z_sp, _, _ = model(x, d_lbl)
                recon, _, q, mu, _ = model(x)
            
            ##
            z_inv = mu
            z_sp = q
            ##
            
            logits = classifier(z_inv, z_sp)
            
            loss_cls = F.cross_entropy(logits, c_lbl)
            
            opt_cls.zero_grad()
            loss_cls.backward()
            opt_cls.step()
            
            total_loss += loss_cls.item()
        
            _, pred = torch.max(logits, dim=1)
            correct += (pred == c_lbl).sum().item()
            total_samples += c_lbl.size(0)
        
        avg_loss = total_loss / len(train_loader)
        acc = 100.0 * correct / total_samples
        print(f"[Epoch {epoch+1}/{num_epochs}] Classifier Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

def eval_class_classifier(model, classifier, test_loader, device='cuda'):
    model.eval()
    classifier.eval()
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for (x, d_lbl, c_lbl) in test_loader:
            x = x.to(device)
            d_lbl = d_lbl.to(device)
            c_lbl = c_lbl.to(device)
            
            # z_inv, _, _, z_sp, _, _ = model(x, d_lbl)
            recon, _, q, mu, _ = model(x)
            z_inv = mu
            z_sp = q
            ##
            logits = classifier(z_inv, z_sp)
            _, pred = torch.max(logits, dim=1)
            
            correct += (pred == c_lbl).sum().item()
            total_samples += c_lbl.size(0)
    
    acc = 100.0 * correct / total_samples
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    joint_vae = JointVAE().to(device)
    domain_cls = DomainClassifier().to(device)

    optimizer_vae = optim.Adam(joint_vae.parameters(), lr=1e-4)
    
    # -----------------------------
    # JVAE 단독 학습
    # -----------------------------
    pretrain_epochs = 200
    for epoch in range(pretrain_epochs):
        vae_loss = train_jvae_only(
            model=joint_vae, 
            train_loader=train_loader, 
            optimizer=optimizer_vae, 
            device=device
        )
        # if (epoch+1) % 10 == 0:
        print(f"[Pretrain Epoch {epoch+1}/{pretrain_epochs}] VAE Loss: {vae_loss:.4f}")
        
        # 재구성 샘플 저장
        # if (epoch+1) % 10 == 0:
        save_reconstructions(joint_vae, test_loader, device, output_dir="recons_pretrain", num_images=5)
        save_samples(joint_vae, device, output_dir="samples_pretrain", num_samples=16)
    
    # -----------------------------
    # Domain Adversarial 학습
    # -----------------------------
    # VAE + DomainClassifier 함께 학습
    optimizer_joint = optim.Adam(
        list(joint_vae.parameters()) + list(domain_cls.parameters()),
        lr=1e-4
    )
    
    adv_epochs = 200
    for epoch in range(adv_epochs):
        vae_loss, dom_loss = train_jvae_domain(
            model=joint_vae,
            domain_cls=domain_cls,
            train_loader=train_loader,
            optimizer=optimizer_joint,
            device=device,
            domain_loss_weight=1.0,  # 조절
            grad_rev_alpha=0.90      # 더 낮게도 가능/조절
        )
        
        # if (epoch+1) % 10 == 0:
        print(f"[Adversarial Epoch {epoch+1}/{adv_epochs}] VAE Loss: {vae_loss:.4f}, Domain Loss: {dom_loss:.4f}")
        
        # 재구성/샘플 저장
        # if (epoch+1) % 10 == 0:
        save_reconstructions(joint_vae, test_loader, device, output_dir="recons_adv", num_images=5)
        save_samples(joint_vae, device, output_dir="samples_adv", num_samples=16)
    
    # -----------------------------
    # Class Label Classifier 학습
    # -----------------------------
    classifier = ClassLabelClassifier(
        z_dim_inv=joint_vae.latent_dim,  # 512
        z_dim_sp=joint_vae.categorical_dim,  # 7
        num_classes=7
    ).to(device)
    
    opt_cls = optim.Adam(classifier.parameters(), lr=1e-4)
    
    # freeze_model=True (JointVAE freeze 후 latent만 뽑아 분류기 학습)
    train_class_classifier(
        model=joint_vae,
        classifier=classifier,
        train_loader=train_loader,
        opt_cls=opt_cls,
        device=device,
        freeze_model=True,
        num_epochs=100
    )
    
    # 테스트셋에서 성능 평가
    eval_class_classifier(joint_vae, classifier, test_loader, device)

if __name__ == "__main__":
    main()