from utils_data import *
from utils_algo import *
from models import *
import argparse, time, os

parser = argparse.ArgumentParser(
	prog='complementary-label learning demo file.',
	usage='Demo with complementary labels.',
	description='A simple demo file with MNIST dataset.',
	epilog='end',
	add_help=True)

parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=5e-5, type=float)
parser.add_argument('-bs', '--batch_size', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-me', '--method', help='method type. non_k_softmax: only equation (7) . w_loss: weighted loss.', choices=['w_loss', 'non_k_softmax'], type=str, required=True)
parser.add_argument('-mo', '--model', help='model name', choices=['linear', 'mlp'], type=str, required=True)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=300)
parser.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-4, type=float)

args = parser.parse_args()

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prepare_mnist_data: for MNIST dataset; prepare_fashion_data: for fashion-MNIST dataset
full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K = prepare_mnist_data(batch_size=args.batch_size)

ordinary_train_loader, complementary_train_loader, ccp = prepare_train_loaders(full_train_loader=full_train_loader, batch_size=args.batch_size, ordinary_train_dataset=ordinary_train_dataset)

if args.model == 'mlp':
    model = mlp_model(input_dim=28*28, hidden_dim=500, output_dim=K)
elif args.model == 'linear':
    model = linear_model(input_dim=28*28, output_dim=K)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
train_accuracy = accuracy_check(loader=train_loader, model=model)
test_accuracy = accuracy_check(loader=test_loader, model=model)
print('Epoch: 0. Tr Acc: {}. Te Acc: {}'.format(train_accuracy, test_accuracy))

save_table = np.zeros(shape=(args.epochs, 4))

for epoch in range(args.epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(complementary_train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = chosen_loss_c(f=outputs, K=K, labels=labels, method=args.method)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss = train_loss + loss.item()
    train_accuracy = accuracy_check(loader=train_loader, model=model)
    test_accuracy = accuracy_check(loader=test_loader, model=model)
    print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr loss: {}.'.format(epoch + 1, train_accuracy, test_accuracy,
                                                                   train_loss / len(complementary_train_loader)))
    save_table[epoch, :] = epoch + 1, train_accuracy, test_accuracy, train_loss / len(complementary_train_loader)


np.savetxt('non_k_sostmax.txt', save_table, delimiter=',', fmt='%1.3f')
