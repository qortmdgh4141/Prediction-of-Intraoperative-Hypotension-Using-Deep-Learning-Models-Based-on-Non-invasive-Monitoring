from .. import *
from src.utils import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from datetime import datetime
from torch.utils.data import DataLoader
from src.train.analyzer import Attn_Saver
import torch

## for AUC score with confidence interval
import numpy as np
from sklearn.model_selection import StratifiedKFold

class TrainWrapper:
    def __init__(self,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader,
                 model: torch.nn.Module,
                 log_path: str,
                 train_settings: dict):

        # INFO: Data Loader settings
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # INFO: Train Settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).type(torch.float64)

        # TODO: Optimization call need to be fixed.
        self.optimizer = call_optimizer(optim_name=train_settings['optimizer'])(self.model.parameters(),
                                                                                lr=train_settings['lr'],
                                                                                momentum=train_settings['momentum'],
                                                                                weight_decay=train_settings['weight_decay'])
        self.loss_fn = call_loss_fn(train_settings['loss_fn'], **train_settings).to(self.device)

        # INFO: Evaluation settings
        dt = datetime.now().strftime("%d-%m-%Y")
        self.log_path = os.path.join(log_path, "{}_{}".format(self.model.name, dt))

        self.model_path = os.path.join(self.log_path, 'check_points')
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_path, 'runs'))
        self.save_count = train_settings['save_count']

        self.best_score = 0
        self.running_loss = 0
        self.correct = 0

        # INFO: Model settings
        self.train_setting = train_settings

        # INFO: Attention evaluations
        if train_settings['save_attn']:
            self.attn = Attn_Saver(self.log_path)
        else:
            self.attn = None

        self.save_hyperparameter(hyperparam=train_settings, name="Train settings")

    def save_hyperparameter(self, hyperparam: dict, name: str):
        with open(os.path.join(self.log_path, "hyperparameters.txt"), "a") as file:
            file.write("{} **********\n".format(name))
            for k, v in hyperparam.items():
                file.write("{}:{}\n".format(k, v))

    def fit(self) -> None:
        """
        Train the model with predefined settings.
        Returns:

        """

        scaler = torch.cuda.amp.GradScaler(enabled=True)

        step_counter = 0
        loss_minima = 1e3

        early_stopping = EarlyStopping(patience=5,
                                       path=os.path.join(self.model_path, 'model.pt'))
        for epoch in tqdm(range(self.train_setting['epochs']), desc="Epochs", position=0):
            correct = []
            total = []
            with tqdm(self.train_loader, desc="Training steps", position=1, postfix={"Train Accuracy": 0}) as pbar:
                for x, y in self.train_loader:
                    self.model.train()
                    input_x = x.to(self.device).type(torch.float64)
                    target_y = y.to(self.device)

                    if self.model.name in ['ValinaLSTM']:
                        hidden = self.model.init_hidden(batch_size=input_x.shape[0], device=self.device)

                    with torch.cuda.amp.autocast(enabled=True):
                        if self.attn:
                            predicted, _, __ = self.model(input_x)
                        elif self.model.name in ['ValinaLSTM']:
                            predicted, _ = self.model(input_x, hidden)
                        else:
                            predicted = self.model(input_x)

                    loss = self.loss_fn(predicted, target_y)

                    # Backpropagation
                    # INFO: Calculate the loss and update the model.
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    # Training evaluation
                    step_counter += 1

                    self.running_loss += loss.item()
                    _, predicted = torch.max(predicted.data, 1)

                    correct.append((predicted == target_y).sum().item())
                    total.append(len(target_y))

                    if step_counter % self.save_count == 0:  # every N step
                        # Loss summary
                        self.running_loss = self.running_loss / self.save_count
                        self.writer.add_scalar('Loss/Train', self.running_loss, step_counter)
                        if self.running_loss < loss_minima:
                            if not os.path.exists(self.model_path):
                                os.makedirs(self.model_path)

                            torch.save({
                                'epoch': epoch,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict()
                            }, os.path.join(self.model_path, 'best-model-{}-{}.pt'.format(epoch, step_counter)))
                            loss_minima = self.running_loss
                        self.running_loss = 0

                        # Validation summary
                        score, acc, _ = self.evaluation(epoch, step_counter)
                        train_acc = sum(correct) / sum(total)

                        pbar.set_postfix({
                            'Train Accuracy': train_acc,
                            'Eval Score': score,
                            'Eval Accuracy': acc
                        })

                    pbar.update(1)

                train_acc = sum(correct) / sum(total)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)

            _, _, valid_loss = self.evaluation(epoch, step_counter)
            early_stopping(valid_loss, model=self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        test_score, test_acc, _ = self.test(self.test_loader)
        pbar.set_postfix({"Train Accuracy": train_acc,
                          "Test Accuracy": test_acc, "Test AUC": test_score})

    def evaluation(self, epoch, step_counter, attention_map=False):
        """
        Args:
            epoch: (int) Represent current epoch. Not epochs in total.

        Returns: None
        """
        score, acc, losses = self._run_eval(self.valid_loader)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if score > self.best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, os.path.join(self.model_path, 'best-model-{}-{}.pt'.format(epoch, step_counter)))
            self.best_score = score

        self.writer.add_scalar('AUC/Valid', score, epoch)
        self.writer.add_scalar('Accuracy/Valid', acc, epoch)
        self.writer.add_scalar('Loss/Valid', losses, epoch)
        return score, acc, losses

    def test(self, data_loader: DataLoader, epoch: int = 0):
        score, test_acc, losses = self._run_eval(data_loader)
        if self.attn:
            self.attn.whole_process()
        self.writer.add_scalar('AUC/Test', score, epoch)
        self.writer.add_scalar('Accuracy/Test', test_acc, epoch)
        self.writer.add_scalar('Loss/Test', losses, epoch)
        return score, test_acc, losses

    def test_ci(self, data_loader: DataLoader, epoch: int = 0):
        mean_score, lower_score, upper_score, test_acc, losses = self._run_eval_ci(data_loader)
        if self.attn:
            self.attn.whole_process()
        self.writer.add_scalar('AUC/Test', mean_score, epoch)
        self.writer.add_scalar('Low_AUC/Test', lower_score, epoch)
        self.writer.add_scalar('Up_AUC/Test', upper_score, epoch)
        self.writer.add_scalar('Accuracy/Test', test_acc, epoch)
        self.writer.add_scalar('Loss/Test', losses, epoch)
        return mean_score, test_acc, losses

    def _run_eval(self, data_loader: DataLoader, with_tqdm: bool = False):
        """
        Returns:
            Tuple[float, float]: AUC Score and Accuracy of test set.
        """
        self.model.eval()
        with torch.no_grad():
            total_correct = []
            total_len = []
            total_loss = []
            list_y = []
            list_y_hat = []
            list_predicted_prob = []

            if with_tqdm:
                pbar = tqdm(data_loader, desc="Evaluation steps")

            # with tqdm(data_loader, desc="Evaluation steps") as pbar:
            for x, y in data_loader:
                x = x.to(self.device).type(torch.float64)
                y = y.to(self.device)

                if self.model.name in ['ValinaLSTM']:
                    hidden = self.model.init_hidden(batch_size=x.shape[0], device=self.device)

                if self.attn:
                    predicted, attention, galr_attention = self.model(x)
                elif self.model.name in ['ValinaLSTM']:
                    predicted, _ = self.model(x, hidden)
                else:
                    predicted = self.model(x)

                total_loss.append(self.loss_fn(predicted, y))

                predict_prob = torch.softmax(predicted, dim=1)
                list_predicted_prob.extend(predict_prob[:, 1].cpu().numpy())

                #_, predicted_y = torch.max(predict_prob, 1)
                predicted_y = (predict_prob[:, 1] > 0.4182).long()

                total_correct.append((predicted_y == y).sum().item())
                total_len.append(len(y))

                list_y += y.detach().cpu().tolist()
                list_y_hat += predicted_y.detach().cpu().tolist()

                accuracy = sum(total_correct)/sum(total_len)

                if with_tqdm:
                    pbar.set_postfix({"Accuracy": accuracy})
                    pbar.update(1)

                if self.attn:
                    self.attn.preparing(y, predicted_y, x, galr_attention, attention)

        total_accuracy = sum(total_correct)/sum(total_len)
        total_score = roc_auc_score(list_y, list_predicted_prob)
        total_loss = sum(total_loss)/sum(total_len)

        if with_tqdm:
            pbar.write("AUC score[{:.2f}] / Accuracy: {:.2f}".format(total_score, total_accuracy))

        return total_score, total_accuracy, total_loss

##implementation for confidence interval
    def _run_eval_ci(self, data_loader: DataLoader, with_tqdm: bool = False):
        """
        Returns:
            Tuple[float, float]: AUC Score and Accuracy of test set.
        """
        self.model.eval()
        with torch.no_grad():
            total_correct = []
            total_len = []
            total_loss = []
            list_y = [] ##true label
            list_y_hat = [] ##predicted label
            list_predicted_prob = [] ##raw prediction probability

            if with_tqdm:
                pbar = tqdm(data_loader, desc="Evaluation steps")

            # with tqdm(data_loader, desc="Evaluation steps") as pbar:
            for x, y in data_loader:
                x = x.to(self.device).type(torch.float64)
                y = y.to(self.device)

                if self.model.name in ['ValinaLSTM']:
                    hidden = self.model.init_hidden(batch_size=x.shape[0], device=self.device)

                if self.attn:
                    predicted, attention, galr_attention = self.model(x)
                elif self.model.name in ['ValinaLSTM']:
                    predicted, _ = self.model(x, hidden)
                else:
                    predicted = self.model(x)

                total_loss.append(self.loss_fn(predicted, y))

                predict_prob = torch.softmax(predicted, dim=1) ##positive class
                ##probability lists...
                list_predicted_prob.extend(predict_prob[:, 1].cpu().numpy())

                _, predicted_y = torch.max(predict_prob, 1)

                total_correct.append((predicted_y == y).sum().item())
                total_len.append(len(y))


                list_y += y.detach().cpu().tolist()
                list_y_hat += predicted_y.detach().cpu().tolist()

                ##accuracy of each fold
                accuracy = sum(total_correct)/sum(total_len)

                if with_tqdm:
                    pbar.set_postfix({"Accuracy": accuracy})
                    pbar.update(1)

                if self.attn:
                    self.attn.preparing(y, predicted_y, x, galr_attention, attention)

        total_accuracy = sum(total_correct)/sum(total_len)
        total_score = roc_auc_score(list_y, list_y_hat) ## predicted_prob ???
        total_loss = sum(total_loss)/sum(total_len)

        skf = StratifiedKFold(n_splits=20, shuffle = True, random_state=42)

        aucs = []

        sensitivities = [] ## remind the definition
        specificities = []

        ppvs = []
        npvs = []



        n = sum(total_len)

        ## we will never use train_idx
        list_y = np.array(list_y)
        list_predicted_prob = np.array(list_predicted_prob)
        for train_idx, valid_idx in skf.split(X=np.zeros(n), y=list_y):
            valid_auc = roc_auc_score(list_y[valid_idx], list_predicted_prob[valid_idx]) ## softly predicted value
            fpr, tpr, thresholds = roc_curve(list_y[valid_idx], list_predicted_prob[valid_idx])

            min_dist = 1
            best_threshold = 0 
            for i in range(len(fpr)):
                fpr_val, tpr_val, threshold = fpr[i], tpr[i], thresholds[i]
                dist = abs(1 - fpr_val - tpr_val)
                if dist < min_dist:
                    min_dist = dist
                    best_threshold = threshold

            y_pred_threshold = (list_predicted_prob[valid_idx] >= best_threshold).astype(float)
            tn, fp, fn, tp = confusion_matrix(list_y[valid_idx], y_pred_threshold).ravel()

            ppv = tp / (tp+fp)
            npv = tn / (fn+tn)
            sensitivity = tp / (tp+fn)
            specificity = tn / (fp+tn)

            aucs.append(valid_auc)
            ppvs.append(ppv)
            npvs.append(npv)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

        aucs = np.array(aucs)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs, ddof=1)
        interval = 1.96 * std_auc / np.sqrt(20)
        lower, upper = mean_auc - interval, mean_auc + interval

        ppvs = np.array(ppvs)
        mean_ppv = np.mean(ppvs)
        std_ppv = np.std(ppvs, ddof=1)
        intervalp = 1.96 * std_ppv / np.sqrt(20)
        lowerp, upperp = mean_ppv - intervalp, mean_ppv + intervalp

        npvs = np.array(npvs)
        mean_npv = np.mean(npvs)
        std_npv = np.std(npvs, ddof=1)
        intervaln = 1.96 * std_npv / np.sqrt(20)
        lowern, uppern = mean_npv - intervaln, mean_npv + intervaln

        sens = np.array(sensitivities)
        mean_sen =np.mean(sens)
        std_sen = np.std(sens,ddof=1)
        intervals = 1.96 * std_sen / np.sqrt(20)
        lowers, uppers = mean_sen - intervals, mean_sen + intervals

        spes = np.array(specificities)
        mean_spes = np.mean(spes)
        std_spes = np.std(spes, ddof=1)
        intervalsp = 1.96 * std_spes / np.sqrt(20)
        lowersp, uppersp = mean_spes - intervalsp, mean_spes + intervalsp


        self.writer.add_scalar('Test/Up_AUC', upper, 0)
        self.writer.add_scalar('Test/Low_AUC', lower, 0)
        self.writer.add_scalar('Test/AUC', mean_auc, 0)

        self.writer.add_scalar('Test/Up_PPV', upperp, 0)
        self.writer.add_scalar('Test/Low_PPV', lowerp, 0)
        self.writer.add_scalar('Test/PPV', mean_ppv, 0)

        self.writer.add_scalar('Test/Up_NPV', uppern, 0)
        self.writer.add_scalar('Test/Low_NPV', lowern, 0)
        self.writer.add_scalar('Test/NPV', mean_npv, 0)

        self.writer.add_scalar('Test/Up_SENS', uppers, 0)
        self.writer.add_scalar('Test/Low_SENS', lowers, 0)
        self.writer.add_scalar('Test/SENSITIVITY', mean_sen, 0)

        self.writer.add_scalar('Test/Up_SP', uppersp, 0)
        self.writer.add_scalar('Test/Low_SP', lowersp, 0)
        self.writer.add_scalar('Test/SPECIFICITY', mean_spes, 0)

        self.writer.add_scalar('Test/Best_Threshold', best_threshold, 0)

        if with_tqdm:
            pbar.write("AUC score[{:.2f}] / Accuracy: {:.2f}".format(total_score, total_accuracy))

        return mean_auc, lower, upper, total_accuracy, total_loss
