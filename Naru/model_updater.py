import time
from abc import abstractmethod, ABC

import numpy as np
import torch

import common
import transformer
from Naru.common_math import Entropy
from utils import dataset_util
from utils.model_util import save_torch_model
from utils.path_util import get_absolute_path


class BaseModelUpdater(ABC):
    def __init__(self, pre_model_path, update_model_path):
        self.pre_model_path = pre_model_path
        self.update_model_path = update_model_path

    @abstractmethod
    def update_model(self):
        pass


class FinetuneModelUpdater(BaseModelUpdater):
    pass


class DDUpModelUpdater(BaseModelUpdater):
    pass


class AdaptModelUpdater(BaseModelUpdater):
    def update_model(self, dataset, pre_model, seed=0):
        torch.manual_seed(0)
        np.random.seed(0)

        # Load data
        table, split_indices = dataset_util.DatasetLoader.load_permuted_dataset(dataset=dataset, permute=False)

        table_bits = Entropy(
            table,
            table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
            [2],
        )

        fixed_ordering = None
        # if args.order is not None:
        #     print("Using passed-in order:", args.order)
        #     fixed_ordering = args.order

        # print(table.data.info())

        """
        Loop over update batches (split indexes) and create new models base on previous model
        """

        # print("split index: {}".format(split_indices))
        for update_step in range(len(split_indices) - 1):
            table_main = table

            if args.heads > 0:
                model = MakeTransformer(
                    cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
                )
                pmodel = MakeTransformer(
                    cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
                )

                pmodel.load_state_dict(torch.load(pre_model))
                pmodel.eval()
            else:
                model = MakeMade(
                    scale=args.fc_hiddens,
                    cols_to_train=table.columns,
                    seed=seed,
                    fixed_ordering=fixed_ordering,
                )
                pmodel = MakeMade(
                    scale=args.fc_hiddens,
                    cols_to_train=table.columns,
                    seed=seed,
                    fixed_ordering=fixed_ordering,
                )
                pmodel.load_state_dict(torch.load(pre_model))
                pmodel.eval()
                model.load_state_dict(torch.load(pre_model))

            mb = ReportModel(model)

            # if False:  # not isinstance(model, transformer.Transformer):
            #     print("Applying InitWeight()")
            #     model.apply(InitWeight)

            if isinstance(model, transformer.Transformer):
                opt = torch.optim.Adam(
                    list(model.parameters()),
                    2e-4,
                    betas=(0.9, 0.98),
                    eps=1e-9,
                )
            else:
                opt = torch.optim.Adam(list(model.parameters()), lr=1e-3)
                decay_rate = 0.98
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=opt, gamma=decay_rate
                )
            bs = args.bs
            log_every = 200

            train_data = common.TableDataset(table_main)
            print("dataset size: {}".format(train_data.size()))

            transfer_data = TransferDataPrepare(
                train_data=train_data, split_indices=split_indices, update_step=update_step
            )

            train_data.tuples = train_data.tuples[
                                split_indices[update_step]: split_indices[update_step + 1]
                                ]

            print(
                "train data shape: {}, transfer data shape:{}".format(
                    train_data.tuples.shape, transfer_data.tuples.shape
                )
            )
            train_losses = []
            train_start = time.time()
            for epoch in range(args.epochs):
                mean_epoch_train_loss = RunAdaptEpoch(
                    "train",
                    model,
                    pmodel,
                    opt,
                    scheduler=lr_scheduler,
                    train_data=train_data,
                    transfer_data=transfer_data,
                    val_data=train_data,
                    omega=0.1,
                    batch_size=bs,
                    epoch_num=epoch,
                    log_every=log_every,
                    table_bits=table_bits,
                )

                if (epoch + 1) % 10 == 0:
                    print(
                        "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                            epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                        )
                    )
                    since_start = time.time() - train_start
                    print("time since start: {:.1f} secs".format(since_start))

                    check_point_path = "./Naru/checkpoints/update_batch_{}_epoch_{}.pt".format(update_step + 1, epoch)
                    absolute_checkpoint_path = get_absolute_path(check_point_path)
                    torch.save(model.state_dict(), absolute_checkpoint_path)
                train_losses.append(mean_epoch_train_loss)

            print("Training done; evaluating likelihood on full data:")
            all_losses = RunEpoch(
                "test",
                model,
                train_data=train_data,
                val_data=train_data,
                opt=None,
                scheduler=lr_scheduler,
                batch_size=1024,
                log_every=500,
                table_bits=table_bits,
                return_losses=True,
            )
            model_nats = np.mean(all_losses)
            model_bits = model_nats / np.log(2)
            model.model_bits = model_bits

            if fixed_ordering is None:
                if seed is not None:
                    PATH = "./models/adapt-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
                        args.dataset,
                        mb,
                        model.model_bits,
                        table_bits[0],
                        args.epochs,
                        seed,
                    )
                else:
                    PATH = "./models/adapt-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-{}.pt".format(
                        args.dataset,
                        mb,
                        model.model_bits,
                        table_bits[0],
                        args.epochs,
                        seed,
                        time.time(),
                    )
            else:
                annot = ""
                if args.inv_order:
                    annot = "-invOrder"

                PATH = "./models/adapt-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-order{}{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                    "_".join(map(str, fixed_ordering)),
                    annot,
                )
            save_torch_model(model, PATH)
            pre_model = PATH
