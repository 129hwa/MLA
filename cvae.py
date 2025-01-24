import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset
from torchvision import transforms
import os
from torchvision.datasets import ImageFolder

# ================================================
# 1. Gradient Reversal Layer
# ================================================
class GradReverse(torch.autograd.Function):
    """
    도메인 분류를 속이기 위한 Gradient Reversal
    """
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

# ================================================
# 2. CVAE Encoder
#    - z_inv: 도메인 불변
#    - z_sp : 도메인 종속 (도메인 라벨 임베딩 포함)
# ================================================
class CVAE_Encoder(nn.Module):
    def __init__(self, 
                 num_domains=4,
                 domain_emb_dim=7,
                 z_dim_inv=32, 
                 z_dim_sp=32, 
                 base_ch=64):
        """
        base_ch: 첫 번째 Conv 채널 수
        """
        super().__init__()
        self.num_domains = num_domains
        self.domain_emb_dim = domain_emb_dim
        self.z_dim_inv = z_dim_inv
        self.z_dim_sp = z_dim_sp
        
        # (1) Domain embedding
        self.domain_emb = nn.Embedding(num_domains, domain_emb_dim)
        
        # (2) Convolution: 224 -> 112 -> 56 -> 28 -> 14 -> 7 (stride=2, 5단계)
        self.encoder_net = nn.Sequential(
            nn.Conv2d(3, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),
            
            nn.Conv2d(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(True),
            
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(True),
            
            nn.Conv2d(base_ch*4, base_ch*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(True),
            
            nn.Conv2d(base_ch*8, base_ch*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(True),
        )
        
        self.final_conv_dim = base_ch*8 * 7 * 7
        
        # (3) z_inv를 위한 mu/logvar
        self.fc_mu_inv = nn.Linear(self.final_conv_dim, z_dim_inv)
        self.fc_logvar_inv = nn.Linear(self.final_conv_dim, z_dim_inv)
        
        # (4) z_sp를 위한 mu/logvar (domain_emb concat)
        #     flattened + domain_emb_dim -> z_dim_sp
        self.fc_mu_sp = nn.Linear(self.final_conv_dim + domain_emb_dim, z_dim_sp)
        self.fc_logvar_sp = nn.Linear(self.final_conv_dim + domain_emb_dim, z_dim_sp)
        
    def forward(self, x, domain_labels):
        """
        x: 이미지 [B, 3, 224, 224]
        domain_labels: [B] (정수, 0~3)
        """
        # 1) 공통 인코딩
        h = self.encoder_net(x)
        h = h.view(h.size(0), -1)
        
        # 2) z_inv
        mu_inv = self.fc_mu_inv(h)
        logvar_inv = self.fc_logvar_inv(h)
        z_inv = self.reparameterize(mu_inv, logvar_inv)
        
        # 3) domain 임베딩
        d_emb = self.domain_emb(domain_labels)
        
        # 4) z_sp
        sp_input = torch.cat([h, d_emb], dim=1)
        mu_sp = self.fc_mu_sp(sp_input)
        logvar_sp = self.fc_logvar_sp(sp_input)
        z_sp = self.reparameterize(mu_sp, logvar_sp)
        
        return z_inv, mu_inv, logvar_inv, z_sp, mu_sp, logvar_sp
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# ================================================
# 3. CVAE Decoder (Conditional)
#    - 입력: z_inv, z_sp, domain_emb
# ================================================
class CVAE_Decoder(nn.Module):
    def __init__(self,
                 num_domains=4,
                 domain_emb_dim=8,
                 z_dim_inv=32,
                 z_dim_sp=32,
                 base_ch=64):
        super().__init__()
        self.num_domains = num_domains
        self.domain_emb_dim = domain_emb_dim
        self.z_dim_inv = z_dim_inv
        self.z_dim_sp = z_dim_sp
        

        self.domain_emb = nn.Embedding(num_domains, domain_emb_dim)
        
        in_dim = z_dim_inv + z_dim_sp + domain_emb_dim
        
        self.fc = nn.Linear(in_dim, base_ch*8*7*7)
        
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(base_ch, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z_inv, z_sp, domain_labels):
        """
        z_inv: [B, z_dim_inv]
        z_sp:  [B, z_dim_sp]
        domain_labels: [B]
        """
        d_emb = self.domain_emb(domain_labels)
        z = torch.cat([z_inv, z_sp, d_emb], dim=1)
        
        h = self.fc(z)
        h = h.view(h.size(0), -1, 7, 7)
        
        x_recon = self.decoder_net(h)
        return x_recon

# ================================================
# 4. Domain Classifier (Adversarial)
#    - 입력: z_inv
#    - 출력: 4개 도메인 로짓
# ================================================
class DomainClassifier(nn.Module):
    def __init__(self, z_dim_inv=32, num_domains=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim_inv, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, num_domains)
        )
        
    def forward(self, z_inv, alpha=1.0):
        # Gradient reversal로 Encoder 학습 시 도메인 분류가 어렵도록
        z_rev = grad_reverse(z_inv, alpha)
        logits = self.net(z_rev)
        return logits

# ================================================
# 5. Loss
# ================================================
def cvae_loss(x, x_recon, 
              mu_inv, logvar_inv,
              mu_sp,  logvar_sp,
              recon_type='mse'):
    """
    - 재구성 손실 (mse 또는 bce)
    - KL Divergence( z_inv + z_sp )
    """
    # (1) Reconstruction
    if recon_type == 'mse':
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    else:
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
    
    # (2) KL
    kld_inv = -0.5 * torch.sum(1 + logvar_inv - mu_inv.pow(2) - logvar_inv.exp(), dim=1).mean()
    kld_sp  = -0.5 * torch.sum(1 + logvar_sp  - mu_sp.pow(2)  - logvar_sp.exp(),  dim=1).mean()
    
    total_loss = recon_loss + (kld_inv + kld_sp)
    return total_loss, recon_loss, (kld_inv + kld_sp)

def domain_classification_loss(logits, domain_labels):
    """
    CrossEntropy Loss
    """
    return F.cross_entropy(logits, domain_labels)

# ================================================
# 6. 학습
# ================================================
def train_epoch(encoder, decoder, domain_classifier,
                dataloader,
                opt_enc, opt_dec, opt_dom,
                alpha=1.0,
                device='cuda'):
    encoder.train()
    decoder.train()
    domain_classifier.train()
    
    total_cvae_loss = 0.0
    total_domain_loss = 0.0
    
    for i, (x, domain_labels, class_labels) in enumerate(dataloader):
        x = x.to(device)
        domain_labels = domain_labels.to(device)
        class_labels = class_labels.to(device)
        
        # ---------------------------
        # (1) Encoder & Decoder update
        # ---------------------------
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        
        # 1) Forward
        z_inv, mu_inv, logvar_inv, z_sp, mu_sp, logvar_sp = encoder(x, domain_labels)
        x_recon = decoder(z_inv, z_sp, domain_labels)
        
        # 2) CVAE Loss
        loss_cvae, rec_loss, kl_loss = cvae_loss(
            x, x_recon, 
            mu_inv, logvar_inv,
            mu_sp, logvar_sp,
            recon_type='mse'
        )
        
        # 3) Domain Classifier (adversarial for encoder)
        domain_logits = domain_classifier(z_inv, alpha=alpha)
        loss_dom_adv = domain_classification_loss(domain_logits, domain_labels)
        
        #   Encoder 입장에서는 도메인 분류가 잘 안 되게! 만들어야 하므로
        #   grad reversal로 자동 부호 반전 -> 여기서는 그대로 더해주면 됨
        loss_enc_total = loss_cvae + loss_dom_adv
        
        loss_enc_total.backward()
        opt_enc.step()
        opt_dec.step()
        
        # ---------------------------
        # (2) Domain Classifier update
        # ---------------------------
        opt_dom.zero_grad()
        # domain classifier는 z_inv 제대로 분류하는 방향으로 학습
        with torch.no_grad():
            # encoder는 업데이트된 상태이므로 다시 forward
            z_inv_after, _, _, _, _, _ = encoder(x, domain_labels)
        domain_logits_after = domain_classifier(z_inv_after, alpha=0.0)
        
        loss_dom = domain_classification_loss(domain_logits_after, domain_labels)
        loss_dom.backward()
        opt_dom.step()
        
        total_cvae_loss += loss_cvae.item()
        total_domain_loss += loss_dom.item()
    
    return total_cvae_loss / len(dataloader), total_domain_loss / len(dataloader)


# ================================================
def main():
    lr = 1e-4
    batch_size = 16
    num_epochs = 100
    alpha = 1.0  # gradient reversal 강도
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class DomainLabeledDataset(Dataset):
        """기존 데이터셋에
        도메인 레이블을 추가하는
        래퍼 클래스"""
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

    # 각 도메인별 데이터셋 생성 (도메인 레이블 추가 버전)
    all_datasets = []
    for domain_idx, domain_name in enumerate(all_domains):
        domain_full_path = os.path.join(data_path, domain_name)
        base_dataset = ImageFolder(domain_full_path, transform=data_transform)
        wrapped_dataset = DomainLabeledDataset(base_dataset, domain_idx)  # 원본 도메인 인덱스 사용
        all_datasets.append(wrapped_dataset)

    # 도메인 분할 (테스트 도메인 제외)
    test_domain_idx = 0
    test_dataset = all_datasets[test_domain_idx]
    train_datasets = [ds for i, ds in enumerate(all_datasets) if i != test_domain_idx]

    # 학습 데이터 결합
    train_dataset = ConcatDataset(train_datasets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

    # (2) 모델 정의
    encoder = CVAE_Encoder(num_domains=4, domain_emb_dim=8, z_dim_inv=32, z_dim_sp=32, base_ch=64).to(device)
    decoder = CVAE_Decoder(num_domains=4, domain_emb_dim=8, z_dim_inv=32, z_dim_sp=32, base_ch=64).to(device)
    domain_classifier = DomainClassifier(z_dim_inv=32, num_domains=4).to(device)
    
    # (3) Optimizer
    opt_enc = optim.Adam(encoder.parameters(), lr=lr)
    opt_dec = optim.Adam(decoder.parameters(), lr=lr)
    opt_dom = optim.Adam(domain_classifier.parameters(), lr=lr)
    
    # (4) 학습
    for epoch in range(num_epochs):
        cvae_loss_avg, dom_loss_avg = train_epoch(
            encoder, decoder, domain_classifier,
            train_loader,
            opt_enc, opt_dec, opt_dom,
            alpha=alpha,
            device=device
        )
        print(f"[Epoch {epoch+1}/{num_epochs}] CVAE Loss: {cvae_loss_avg:.4f}, Domain Loss: {dom_loss_avg:.4f}")

    # (5) 모델 저장
    # torch.save(encoder.state_dict(), "encoder.pth")
    # torch.save(decoder.state_dict(), "decoder.pth")
    # torch.save(domain_classifier.state_dict(), "domain_classifier.pth")

if __name__ == "__main__":
    main()
