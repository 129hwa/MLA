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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image

########################################
# 1. Gradient Reversal for Domain Adversarial
########################################
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

########################################
# 2. FactorVAE_InvSp
#    - z_inv, z_sp 분리
########################################
class FactorVAE_InvSp(nn.Module):
    """
    FactorVAE 구조 + (z_inv, z_sp) 분리

    - z_dim_inv: 도메인 불변 latent 차원
    - z_dim_sp : 도메인 종속 latent 차원
    - z_dim_total = z_dim_inv + z_dim_sp
    - gamma     : TC( total correlation ) 손실 계수
    """
    def __init__(self, 
                 in_channels=3,
                 z_dim_inv=32,
                 z_dim_sp=32,
                 gamma=5.0):
        super().__init__()

        self.z_dim_inv = z_dim_inv
        self.z_dim_sp = z_dim_sp
        self.z_dim_total = z_dim_inv + z_dim_sp
        self.gamma = gamma

        # ------------------
        #   Encoder (CNN) -> res로 바꿔라
        # ------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        # Flatten dimension: 512 * 7 * 7 = 25088
        self.flatten_dim = 512 * 7 * 7

        # ------------------
        #  (z_inv) branch
        # ------------------
        self.fc_mu_inv = nn.Linear(self.flatten_dim, z_dim_inv)
        self.fc_logvar_inv = nn.Linear(self.flatten_dim, z_dim_inv)

        # ------------------
        #  (z_sp) branch
        # ------------------
        self.fc_mu_sp = nn.Linear(self.flatten_dim, z_dim_sp)
        self.fc_logvar_sp = nn.Linear(self.flatten_dim, z_dim_sp)

        # ------------------
        #   Decoder
        # ------------------
        self.decoder_input = nn.Linear(self.z_dim_total, 512*7*7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # ------------------
        #  Discriminator (TC)
        # ------------------
        self.discriminator = nn.Sequential(
            nn.Linear(self.z_dim_total, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2)
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)

        mu_inv = self.fc_mu_inv(h)
        logvar_inv = self.fc_logvar_inv(h)
        
        mu_sp = self.fc_mu_sp(h)
        logvar_sp = self.fc_logvar_sp(h)

        return mu_inv, logvar_inv, mu_sp, logvar_sp

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_inv, z_sp):
        z_cat = torch.cat([z_inv, z_sp], dim=1)
        h = self.decoder_input(z_cat)
        h = h.view(-1, 512, 7, 7)
        x_recon = self.decoder(h)
        return x_recon

    def permute_latent(self, z: torch.Tensor):
        """
        배치 내에서 z를 랜덤 셔플(각 sample별로 z dimension 교란)
        FactorVAE TC 계산 시 real vs perm 구분
        """
        B, D = z.size()
        inds = torch.cat([(D * i) + torch.randperm(D) for i in range(B)])
        return z.view(-1)[inds].view(B, D)

    def forward(self, x):
        """
        return:
          x_recon, x_in, (z_inv, mu_inv, logvar_inv), (z_sp, mu_sp, logvar_sp), z_cat
        """
        mu_inv, logvar_inv, mu_sp, logvar_sp = self.encode(x)
        z_inv = self.reparameterize(mu_inv, logvar_inv)
        z_sp  = self.reparameterize(mu_sp,  logvar_sp)

        x_recon = self.decode(z_inv, z_sp)
        z_cat = torch.cat([z_inv, z_sp], dim=1)

        return x_recon, x, (z_inv, mu_inv, logvar_inv), (z_sp, mu_sp, logvar_sp), z_cat

    def loss_function(self, *args, **kwargs):
        """
        args:
          x_recon, x_in, (z_inv, mu_inv, logvar_inv), (z_sp, mu_sp, logvar_sp), z_cat
        kwargs:
          optimizer_idx : 0(VAE), 1(Discriminator)
          M_N (kld_weight)
        """
        x_recon, x_in, z_inv_tuple, z_sp_tuple, z_cat = args
        z_inv, mu_inv, logvar_inv = z_inv_tuple
        z_sp,  mu_sp,  logvar_sp  = z_sp_tuple

        optimizer_idx = kwargs.pop('optimizer_idx')
        kld_weight = kwargs.get('M_N', 1.0)

        # 1) VAE Update
        if optimizer_idx == 0:
            recons_loss = F.mse_loss(x_recon, x_in, reduction='mean')
            # KL(inv) + KL(sp)
            kld_inv = -0.5 * torch.sum(1 + logvar_inv - mu_inv**2 - logvar_inv.exp(), dim=1).mean()
            kld_sp  = -0.5 * torch.sum(1 + logvar_sp  - mu_sp**2  - logvar_sp.exp(),  dim=1).mean()
            kld_total = kld_inv + kld_sp

            # VAE TC loss
            D_z = self.discriminator(z_cat)  # [B, 2]
            vae_tc_loss = (D_z[:, 0] - D_z[:, 1]).mean()

            total_loss = recons_loss + kld_weight*kld_total + self.gamma*vae_tc_loss
            return {'loss': total_loss}

        # 2) Discriminator (TC) Update
        elif optimizer_idx == 1:
            z_cat_detach = z_cat.detach()
            # real
            D_z = self.discriminator(z_cat_detach)
            # perm
            z_perm = self.permute_latent(z_cat_detach)
            D_z_perm = self.discriminator(z_perm)

            B = z_cat_detach.size(0)
            device = z_cat_detach.device
            # label: real=0, perm=1
            D_tc_loss = 0.5 * (
                F.cross_entropy(D_z,       torch.zeros(B, dtype=torch.long, device=device)) +
                F.cross_entropy(D_z_perm,  torch.ones(B,  dtype=torch.long, device=device))
            )
            return {'loss': D_tc_loss}

    def sample(self, num_samples: int, device='cuda'):
        """
        무작위 (z_inv, z_sp)로부터 이미지 샘플링
        """
        z_inv = torch.randn(num_samples, self.z_dim_inv, device=device)
        z_sp  = torch.randn(num_samples, self.z_dim_sp,  device=device)
        x_samples = self.decode(z_inv, z_sp)
        return x_samples

    def generate(self, x):
        """
        x -> (z_inv, z_sp) -> x_recon
        """
        return self.forward(x)[0]  # x_recon

########################################
# 3. Domain Classifier (z_inv -> domain)
########################################
class DomainClassifier(nn.Module):
    """
    z_inv만 입력 -> 도메인 예측 (P/A/C/S)
    Gradient Reversal로 도메인 정보 최소화
    """
    def __init__(self, z_dim_inv=32, num_domains=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim_inv, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, z_inv, alpha=1.0):
        # gradient reversal로 도메인 분류를 어렵게
        z_rev = grad_reverse(z_inv, alpha)
        return self.net(z_rev)

########################################
# 4. Class Label Classifier
#    (z_inv, z_sp) -> class label
########################################
class ClassLabelClassifier(nn.Module):
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

########################################
# 5. 학습 (FVAE + Domain)
########################################
def train_fvae_domain(model, domain_cls, train_loader,
                      optimizer_vae, optimizer_d, optimizer_dom,
                      device='cuda', kld_weight=1.0):
    """
    FactorVAE + DomainClassifier 학습
      1) VAE (optimizer_idx=0) + DomainClassifier
      2) Discriminator (optimizer_idx=1)
    """
    model.train()
    domain_cls.train()

    total_vae_loss = 0.0
    total_dom_loss = 0.0
    total_disc_loss = 0.0

    for (x, d_lbl, _) in train_loader:
        x = x.to(device)
        d_lbl = d_lbl.to(device)

        # -------------------------
        # (1) VAE + Domain update
        # -------------------------
        # forward
        x_recon, x_in, (z_inv, mu_inv, logvar_inv), (z_sp, mu_sp, logvar_sp), z_cat = model(x)

        # FactorVAE loss (optimizer_idx=0)
        loss_dict_vae = model.loss_function(
            x_recon,
            x_in,
            (z_inv, mu_inv, logvar_inv),
            (z_sp, mu_sp, logvar_sp),
            z_cat,
            optimizer_idx=0,
            M_N=kld_weight
        )
        vae_loss = loss_dict_vae['loss']

        # Domain Classification (z_inv -> domain)
        dom_logits = domain_cls(z_inv)
        dom_loss = F.cross_entropy(dom_logits, d_lbl)

        total_loss = vae_loss + dom_loss

        optimizer_vae.zero_grad()
        optimizer_dom.zero_grad()
        total_loss.backward()
        optimizer_vae.step()
        optimizer_dom.step()

        total_vae_loss += vae_loss.item()
        total_dom_loss += dom_loss.item()

        # -------------------------
        # (2) Discriminator update
        # -------------------------
        # 다시 forward (batch 동일)
        x_recon_d, x_in_d, (z_inv_d, _, _), (z_sp_d, _, _), z_cat_d = model(x)
        loss_dict_d = model.loss_function(
            x_recon_d, x_in_d, (z_inv_d, None, None), (z_sp_d, None, None), z_cat_d,
            optimizer_idx=1,
            M_N=kld_weight
        )
        disc_loss = loss_dict_d['loss']

        optimizer_d.zero_grad()
        disc_loss.backward()
        optimizer_d.step()

        total_disc_loss += disc_loss.item()

    n = len(train_loader)
    return (total_vae_loss / n, total_dom_loss / n, total_disc_loss / n)

########################################
# 6. Class Label Classifier 학습
########################################
def train_class_classifier(model, classifier, train_loader, opt_cls,
                           device='cuda', freeze_model=True, num_epochs=5):
    """
    FactorVAE_InvSp로부터 (z_inv, z_sp)를 추출해
    class label 분류기 학습
    """
    if freeze_model:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for (x, d_lbl, c_lbl) in train_loader:
            x = x.to(device)
            c_lbl = c_lbl.to(device)

            # encoder -> (z_inv, z_sp)
            with torch.no_grad() if freeze_model else torch.enable_grad():
                x_recon, x_in, (z_inv, _, _), (z_sp, _, _), _ = model(x)

            # 분류기 forward
            logits = classifier(z_inv, z_sp)
            loss_cls = F.cross_entropy(logits, c_lbl)

            opt_cls.zero_grad()
            loss_cls.backward()
            opt_cls.step()

            total_loss += loss_cls.item()
            preds = logits.argmax(dim=1)
            correct += (preds == c_lbl).sum().item()
            total_samples += c_lbl.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total_samples
        print(f"Epoch[{epoch+1}/{num_epochs}] Classifier Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

########################################
# 7. 테스트 시 정확도 측정
########################################
def eval_class_classifier(model, classifier, test_loader, device='cuda'):
    model.eval()
    classifier.eval()

    correct = 0
    total_samples = 0
    with torch.no_grad():
        for (x, d_lbl, c_lbl) in test_loader:
            x = x.to(device)
            c_lbl = c_lbl.to(device)

            x_recon, x_in, (z_inv, _, _), (z_sp, _, _), _ = model(x)
            logits = classifier(z_inv, z_sp)
            preds = logits.argmax(dim=1)

            correct += (preds == c_lbl).sum().item()
            total_samples += c_lbl.size(0)

    acc = 100.0 * correct / total_samples
    print(f"[eval_class_classifier] Accuracy: {acc:.2f}%")
    return acc

########################################
# 8. 샘플/재구성 이미지 저장
########################################
def save_samples(model, device, output_dir='samples', num_samples=16):
    """
    무작위 latent(z_inv,z_sp) -> 이미지 생성
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device=device)

    file_path = os.path.join(output_dir, "random_samples.png")
    save_image(samples, file_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"[save_samples] Saved random samples to '{file_path}'")

def save_reconstructions(model, loader, device, output_dir='recons', num_images=5):
    """
    실제 배치 일부 -> 재구성 결과
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    images, domain_lbl, class_lbl = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        x_recon, _, _, _, _ = model(images)

    for i in range(num_images):
        save_image(images[i],
                   os.path.join(output_dir, f"original_{i}.png"),
                   normalize=True, value_range=(-1,1))
        save_image(x_recon[i],
                   os.path.join(output_dir, f"reconstructed_{i}.png"),
                   normalize=True, value_range=(-1,1))

    print(f"[save_reconstructions] Saved {num_images} pairs of original/reconstructed images to '{output_dir}'")


########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (2) FactorVAE_InvSp, DomainClassifier
    model = FactorVAE_InvSp(in_channels=3, z_dim_inv=32, z_dim_sp=32, gamma=40.0).to(device)
    domain_cls = DomainClassifier(z_dim_inv=32, num_domains=4).to(device)

    # (3) Optimizers
    #  - VAE (encoder+decoder)
    vae_params = (list(model.encoder.parameters()) +
                  list(model.fc_mu_inv.parameters()) +
                  list(model.fc_logvar_inv.parameters()) +
                  list(model.fc_mu_sp.parameters()) +
                  list(model.fc_logvar_sp.parameters()) +
                  list(model.decoder_input.parameters()) +
                  list(model.decoder.parameters()))
    optimizer_vae = optim.Adam(vae_params, lr=1e-4)

    #  - Discriminator
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=1e-5)

    #  - DomainClassifier
    optimizer_dom = optim.Adam(domain_cls.parameters(), lr=1e-4)

    # (4) VAE+Domain+Discriminator 모두 학습
    num_epochs = 100
    for epoch in range(num_epochs):
        vae_loss_avg, dom_loss_avg, disc_loss_avg = train_fvae_domain(
            model, domain_cls, train_loader,
            optimizer_vae, optimizer_d, optimizer_dom,
            device=device,
            kld_weight=1.5
        )
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"VAE Loss: {vae_loss_avg:.4f} | "
              f"Dom Loss: {dom_loss_avg:.4f} | "
              f"Disc Loss: {disc_loss_avg:.4f}")

        # 재구성/샘플링 결과를 저장
        save_reconstructions(model, test_loader, device, output_dir="recons", num_images=2)
        save_samples(model, device, output_dir="samples", num_samples=4)

    # (5) Class Label Classifier 학습
    classifier = ClassLabelClassifier(z_dim_inv=32, z_dim_sp=32, num_classes=7).to(device)
    opt_cls = optim.Adam(classifier.parameters(), lr=1e-4)

    train_class_classifier(model, classifier, train_loader, opt_cls, device, freeze_model=True, num_epochs=50)
    eval_class_classifier(model, classifier, test_loader, device)

if __name__ == "__main__":
    main()
