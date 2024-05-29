import torch
import torch.nn as nn
import torch.optim as optim
from general_utils import AttrDict
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator
import numpy as np
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
import os
save_dir = "model_weights"
os.makedirs(save_dir, exist_ok=True)

# Check if the MPS (Metal Performance Shaders) backend is available
if torch.backends.mps.is_available():
    device = torch.device('cpu')
    print("MPS is available")
else:
    device = torch.device('cpu')
    print("MPS is not available")

class MainLoop:
    def __init__(self):
        pass

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.flatten(start_dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = x.view(64, 1, 1)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))
        return x

class RewardHead(nn.Module):
    def __init__(self, input_dim):
        super(RewardHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FullModel(nn.Module):
    def __init__(self, conditioning_frames=3):
        super(FullModel, self).__init__()
        self.encoder = Encoder()
        self.mlp = MLP(input_dim=192, hidden_dim=32, output_dim=32)
        self.predictor = Predictor(input_dim=32, hidden_dim=32)
        self.reward_head = RewardHead(input_dim=32)
        self.decoder = Decoder()
        self.lstm_input_list = []
        self.conditioning_frames = conditioning_frames

    def forward(self, x):
        z = self.encoder(x)
        mlp_out = self.mlp(z)
        mlp_output = mlp_out.clone().detach()
        self.lstm_input_list.append(mlp_output)

        if len(self.lstm_input_list) > self.conditioning_frames:
            self.lstm_input_list.pop(0)
            lstm_in = torch.stack(self.lstm_input_list).unsqueeze(0)
            lstm_out = self.predictor(lstm_in)
            rewards = self.reward_head(lstm_out)
            rewards = rewards.flatten(start_dim=0)
            return rewards, z

        return torch.zeros((self.conditioning_frames)), z
    
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    import cv2
    from general_utils import make_image_seq_strip
    from sprites_datagen.rewards import ZeroReward, VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward
    
    spec = AttrDict(
        resolution=64,
        max_seq_len=500,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=2,
        rewards=[VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )
    
    gen = DistractorTemplateMovingSpritesGenerator(spec)

    prediction_horizon = 4

    
    model = FullModel(conditioning_frames=prediction_horizon).to(device)
    model.apply(initialize_weights)
    decoder = Decoder().to(device)
    decoder.apply(initialize_weights)
    model_optimizer = optim.Adam(model.parameters(),  lr=0.0001, betas=(0.9, 0.999))
    reconstructor_optimizer = optim.Adam(decoder.parameters(),  lr=0.0002, betas=(0.9, 0.999))

    
    # Load the state dictionaries
    try:
        checkpoint = torch.load(os.path.join(save_dir, 'checkpoint_epoch_d68.pth'))  # Replace X with the specific epoch number
        model.load_state_dict(checkpoint['model_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    except:
        print("Error loading model weights")

    reconstruction_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()

    epoch_loss_list = []

    for epoch in range(100):
        traj = gen.gen_trajectory()

        torch_images = torch.stack([torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in traj.images]).to(device) / 255
        loss_list = []
        reward_tuple = np.sqrt((np.array(traj.rewards["agent_x"])-np.array(traj.rewards["target_x"]))**2 +(np.array(traj.rewards["agent_y"])-np.array(traj.rewards["target_y"]))**2)
               
        for timestep in range(0, len(torch_images) - 3 - prediction_horizon):
            model_optimizer.zero_grad()
            reconstructor_optimizer.zero_grad()
            observation = torch_images[timestep:timestep+3]
            # t t+1 t+2
            pred_rewards, encoded = model(observation)
            reward_loss = None
            actual_rewards = None

            if not torch.all(torch.eq(pred_rewards, torch.zeros_like(pred_rewards))):
                rewards = reward_tuple[timestep+3:timestep+3+prediction_horizon]
                #print(rewards)
                # t+3 t+4 t+5 t+6
                actual_rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                reward_loss = reward_loss_fn(pred_rewards, actual_rewards)
                reward_loss.backward()

            encoded_0 = encoded[0].clone().detach()
            decoded_image = decoder(encoded_0)
            reconstruction_loss = reconstruction_loss_fn(decoded_image, torch_images[timestep])
            reconstruction_loss.backward()

            if timestep % 100 == 0:
                if False:
                    image = decoded_image.squeeze().detach().cpu().numpy() * 255
                    original_image = torch_images[timestep+3].squeeze().detach().cpu().numpy() *255
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                    image = cv2.resize(image, (512, 512))
                    original_image = cv2.resize(original_image, (512, 512))

                    if actual_rewards is not None:
                        pred_rewards = pred_rewards.cpu().clone().detach().numpy()
                        x = int(pred_rewards[0]*500)
                        #print(x)
                        #cv2.line(original_image, (x, 0), (x, 500), (255, 0, 0), 1)

                    stacked_image = np.hstack((image, original_image))
                    cv2.imshow("Reconstructed Image", stacked_image)
                    cv2.waitKey(1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                print(f"Epoch {epoch}, timestep {timestep}, model_loss: {reward_loss}, reconstructor_loss: {reconstruction_loss}")

            model_optimizer.step()
            reconstructor_optimizer.step()
            if reward_loss is not None:
                loss_list.append((reward_loss.cpu().clone().detach(), reconstruction_loss.cpu().clone().detach()))

        epoch_reward_loss = sum([l[0] for l in loss_list]) / len(loss_list)
        epoch_reconstruction_loss = sum([l[1] for l in loss_list]) / len(loss_list)
        epoch_loss_list.append((epoch_reward_loss, epoch_reconstruction_loss))
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'model_optimizer_state_dict': model_optimizer.state_dict(),
        'reconstructor_optimizer_state_dict': reconstructor_optimizer.state_dict(),
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))


    
    plt.plot([l[0] for l in epoch_loss_list], label='Reward Loss')
    plt.plot([l[1] for l in epoch_loss_list], label='Reconstruction Loss')
    plt.legend()
    plt.show()