import torch
import torch.nn as nn
import torch.nn.functional as F
import model
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from numpy import array
import os
import random
import torch.optim as optim
import shutil

# 加载预训练模型
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 加载运算设备并将预训练模型加载至指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

end_of_text = '<|endoftext|>'

# 加载wte 和 wpe
wte = model.model.state_dict()['transformer.wte.weight']  # [vocab_size, n_embed]
wpe = model.model.state_dict()['transformer.wpe.weight']  # [vocab_size, n_embed]

# 加载预训练模型配置
gpt2_config = GPT2Config()

# 数据路径
path = './data'

# 模型超参数
epoch_num = 1000
lr = 1e-5
batch_size = 1

best_test_loss = float("inf")

# 模型保存路径
save_dir = './models/'

# 实例化交叉熵损失
ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')


def preprocess():
    data_loader = []
    files = os.listdir(path)

    for file in files:
        fo = open(os.path.join(path, file), 'r', encoding='utf-8')
        text = fo.read()
        context = ''
        index = 0
        for line in text.split('\n'):

            if 'subject:DE' not in line and 'dialog:' not in line:
                if index < len(text.split('\n'))-1:
                    context += line.lstrip(' ').rstrip(' ').rstrip('\n') + ' '

                else:
                    context += line.lstrip(' ').rstrip(' ').rstrip('\n') + end_of_text
            index += 1
        data = tokenizer(context)['input_ids']
        data_loader.append(data)

        fo.close()

    random.shuffle(data_loader)

    data_loader = [[data_loader[j] for j in range(i-batch_size, i)]
                   for i in range(batch_size, len(data_loader)+1, batch_size)]
    train_loader = data_loader[0:int(len(data_loader)*0.7)]
    test_loader = data_loader[int(len(data_loader)*0.7):]

    return train_loader, test_loader


# 模型类
class VAE(nn.Module):
    def __init__(self, h_dim=gpt2_config.n_embd, z_dim=int(gpt2_config.n_embd/2)):
        super(VAE, self).__init__()

        # encoder part
        self.encoder = model

        for params in self.encoder.parameters():
            params.requires_grad = False
        # self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_sigma

        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim)
        # self.fc5 = nn.Linear(h_dim, input_dim)
        self.decoder = model

        for params in self.decoder.parameters():
            params.requires_grad = False

    def forward(self, x):
        mu, log_sigma = self.encode(x)

        sampled_z = self.reparameterzie(mu, log_sigma)
        sampled_z = torch.reshape(input=sampled_z, shape=[sampled_z.shape[0], 1, sampled_z.shape[-1]])
        res = self.decode(x=sampled_z, text_ids=x)

        return res, mu, log_sigma

    def encode(self, x):
        """
        encoding part.
        :param x: input_ids
        :return: mu and log_(sigma**2)
        """
        # h = F.relu(self.fc1(x))
        output = self.encoder(input_ids=x, output_hidden_states=True)  # x [batch_size, sequence]
        # print(output.hidden_states[-1].shape)
        h = output.hidden_states[-1][:, -1, :]  # h [batch_size, n_h] output_hidden_states[-1] [batch_size, sequence, n_h]
        # print(h.shape)
        mu = self.fc2(h)  # [batch_size, n_z/z_dim]
        # print(mu.shape)
        log_sigma = self.fc3(h)  # estimate log(sigma**2) actually [batch_size, n_z/z_dim]
        # print(log_sigma.shape)
        return mu, log_sigma

    def reparameterzie(self, mu, log_sigma):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_sigma:
        :return: sampled z
        """

        std = torch.exp(log_sigma * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, x, text_ids):
        """
        Given a sampled z, decode it back to image
        :param x:
        :param text_ids:
        :return:
        """
        h = F.relu(self.fc4(x))  # [batch_size, 1, n_h]  x [batch_size, 1, n_z] n_h=h_dim=n_embed n_z=z_dim
        # print(h)
        # print(text_ids)
        text_ids_shape = text_ids.shape
        text_ids = torch.reshape(input=text_ids, shape=[text_ids_shape[0]*text_ids_shape[-1]])
        # print(text_ids)
        text_wtes = torch.index_select(input=wte, dim=0, index=text_ids)  # [batch_size*sequence, n_embed]
        text_wtes = torch.reshape(input=text_wtes, shape=[text_ids_shape[0], text_ids_shape[-1], wte.shape[-1]])  # [batch_size, sequence, n_embed]
        text_wpes = torch.index_select(input=wpe, dim=0, index=torch.from_numpy(array(list(range(0, text_ids_shape[-1]))*text_ids_shape[0])).to(device))  # [batch_size*sequence, n_embed]
        text_wpes = torch.reshape(input=text_wpes, shape=[text_ids_shape[0], text_ids_shape[-1], wpe.shape[-1]]) # [batch_size, sequence, n_embed]
        # print(text_wtes)
        # print(text_wpes)
        # print(text_wpes.shape)
        inputs_embeds = torch.cat(tensors=(h, text_wtes + text_wpes), dim=1)  # [batch_size, sequence, n_embed]
        output = self.decoder(inputs_embeds=inputs_embeds)
        res = output.logits  # res=logits [batch_size, sequence, vocab_size]
        # res = F.sigmoid(self.fc5(h))
        return res


# 保存模型方法
def save_checkpoint(state, is_best, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_file = os.path.join(save_dir, 'checkpoint.pth')
    best_file = os.path.join(save_dir, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


# loss
def loss_func(recons_x, inputs, mu, log_sigma):
    """

    :param recons_x:
    :param inputs:
    :param mu:
    :param log_sigma:
    :return:
    """
    recons_loss = ce_loss(input=recons_x, target=inputs)

    divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1. - log_sigma)

    loss = recons_loss + divergence
    return loss, divergence, recons_loss


def train():
    train_loader = preprocess.train_loader
    test_loader = preprocess.test_loader

    my_vae = VAE.VAE()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, my_vae.parameters()), lr=lr)
    my_vae.to(device)

    for epoch in range(epoch_num):
        for idx, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            # Remember to deploy the input data on GPU
            input_ids = torch.from_numpy(array(data)).to(device)  # [batch_size, sequence]

            # forward
            res, mu, log_sigma = my_vae.forward(x=input_ids)

            # loss
            logits_shape = res.shape
            logits = torch.reshape(input=res[:, 0:-1, :],
                                   shape=[logits_shape[0] * (logits_shape[1] - 1),
                                          logits_shape[2]])  # [batch_size, sequence, vocab_size]
            input_ids = torch.reshape(input=input_ids, shape=[input_ids.numel()])
            loss, recons_loss, kl_loss = loss_func(recons_x=logits, inputs=input_ids, mu=mu, log_sigma=log_sigma)

            # zero out the paramter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # print statistics every 70 batches
            if (idx + 1) % 70 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f} Total loss {:.4f}"
                      .format(epoch + 1, epoch_num, idx + 1, len(train_loader), recons_loss.item(),
                              kl_loss.item(), loss.item()))

            # if idx == 0:
                # pass
                # visualize reconstructed result at the beginning of each epoch

        # test
        test_avg_loss = 0.0
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                input_ids = torch.from_numpy(array(data)).to(device)
                # forward
                res, mu, log_sigma = my_vae.forward(x=input_ids)

                # loss
                logits_shape = res.shape
                logits = torch.reshape(input=res[:, 0:-1, :],
                                       shape=[logits_shape[0] * (logits_shape[1] - 1),
                                              logits_shape[2]])  # [batch_size, sequence, vocab_size]
                input_ids = torch.reshape(input=input_ids, shape=[input_ids.numel()])
                loss, recons_loss, kl_loss = loss_func(recons_x=logits, inputs=input_ids, mu=mu, log_sigma=log_sigma)

                test_avg_loss += loss

            test_avg_loss /= len(test_loader)
            print('{}, Epoch[{}/{}], avg_loss: {:.4f}'.format('test', epoch+1, epoch_num, test_avg_loss))

            # save model
            global best_test_loss
            is_best = test_avg_loss < best_test_loss
            best_test_loss = min(test_avg_loss, best_test_loss)
            save_checkpoint({
                'epoch': epoch,
                'best_test_loss': best_test_loss,
                'state_dict': my_vae.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, save_dir)

        """ 
           # we randomly sample some images' latent vectors from its distribution
           z = torch.randn(args.batch_size, args.z_dim).to(device)
           random_res = myVAE.decode(z).view(-1, 1, 28, 28)
           save_image(random_res, "./%s/random_sampled-%d.png" % (args.result_dir, epoch + 1))
        """


if __name__ == '__main__':
    train()






