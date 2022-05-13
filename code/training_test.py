import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from util import get_all_texts
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English to German model
model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe').to(device)



def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%1000 == 999:
            last_loss = running_loss/1000
            print(f"  batch {i+1} loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/traing", last_loss, tb_x)
            running_loss = 0.

    return last_loss

training_set, _, validation_set = get_all_texts("EMEA", "en")
training_loader = torch.utils.data.DataLoader(training_set, batch_size=5,
                                              shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=5,
                                                shuffle=False)

print(f"Training set has {len(training_set)} instances")
print(f"Validation set has {len(validation_set)} instances")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001,
                             betas=(0.9, 0.98))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/fasihon_trainer_{timestamp}")
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

# for epoch in range(EPOCHS):
#     print(f"EPOCH: {epoch_number+1}")
#
#     model.train(True)
#     avg_loss = train_one_epoch(epoch_number, writer)
#     model.train(False)
#
#     running_vloss = 0.0
#     for i, vdata in enumerate(validation_loader):
#         vinputs, vlabels = vdata
#         voutputs = model(vinputs)
#         vloss = loss_fn(voutputs, vlabels)
#         running_vloss += vloss
#
#     avg_vloss = running_vloss / (i + 1)
#     print(f"LOSS train {avg_loss} valid {avg_vloss}")
#
#     writer.add_scalars("Training vs. Validation Loss",
#                        {"Training": avg_loss, "Validation": avg_vloss},
#                        epoch_number + 1)
#     writer.flush()
#
#     if avg_loss < best_vloss:
#         best_vloss = avg_vloss
#         model_path = f"model_{timestamp}_{epoch_number}"
#         torch.save(model.state_dict(), model_path)
#
#     epoch_number += 1
