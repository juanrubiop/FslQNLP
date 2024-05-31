from lambeq import BobcatParser, AtomicType, SpacyTokeniser, Rewriter
import numpy as np

from lambeq import TketModel, QuantumTrainer, SPSAOptimizer,remove_cups

#from lambeq import remove_cups

from pytket.extensions.qiskit import AerBackend

import matplotlib.pyplot as plt

from lambeq import AtomicType,BinaryCrossEntropyLoss, Dataset

from lambeq import NumpyModel

from lambeq import IQPAnsatz,Sim15Ansatz

import datetime

from utils.FslAnsatz import FslSim15Ansatz, FslStronglyEntanglingAnsatz, FslBaseAnsatz

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = BobcatParser(verbose='text')
tokeniser = SpacyTokeniser()


def load_data():
    preq_embeddings={}
    with open("resources\embeddings\wikipedia_glove\glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            preq_embeddings[word] = vector
    return preq_embeddings

def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences


def generate_diagrams(train_data,dev_data,test_data):
    raw_train_tokens = tokeniser.tokenise_sentences(train_data)
    raw_dev_tokens = tokeniser.tokenise_sentences(dev_data)
    raw_test_tokens = tokeniser.tokenise_sentences(test_data)

    raw_train_diagrams = parser.sentences2diagrams(raw_train_tokens,tokenised=True)
    raw_dev_diagrams = parser.sentences2diagrams(raw_dev_tokens,tokenised=True)
    raw_test_diagrams = parser.sentences2diagrams(raw_test_tokens,tokenised=True)


    train_diagrams=raw_train_diagrams
    dev_diagrams=raw_dev_diagrams
    test_diagrams=raw_test_diagrams

    train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
    dev_diagrams = [remove_cups(diagram) for diagram in raw_dev_diagrams]
    test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]
    
    return train_diagrams, dev_diagrams, test_diagrams


new_train_labels, new_train_data = read_data('resources\dataset\\new_mc_clean_all_data.txt')

indices = np.arange(len(new_train_data))
np.random.shuffle(indices)
new_train_data =np.array(new_train_data)[indices]
new_train_labels=np.array(new_train_labels)[indices]

train_labels=new_train_labels[0:2953].tolist()
train_data=new_train_data[0:2953].tolist()
test_labels=new_train_labels[2954:5906].tolist()
test_data=new_train_data[2954:5906].tolist()
dev_labels=new_train_labels[5907:].tolist()
dev_data=new_train_data[5907:].tolist()

TESTING=True

if TESTING:
    train_labels, train_data = train_labels[:2], train_data[:2]
    dev_labels, dev_data = dev_labels[:2], dev_data[:2]
    test_labels, test_data = test_labels[:2], test_data[:2]
    EPOCHS = 1

train_diagrams, dev_diagrams, test_diagrams=generate_diagrams(train_data=train_data,dev_data=dev_data,test_data=test_data)

def create_circuits(map,n_layers,ansatz_string,preq_embeddings):
    match ansatz_string:
        case "FslBase":
            ansatz = FslBaseAnsatz(preq_embeddings,map, n_layers=n_layers)
        case "FslSim15":
            ansatz = FslSim15Ansatz(preq_embeddings,map, n_layers=n_layers)  
        case "Sim15Ansatz":
            ansatz = Sim15Ansatz(map,n_layers=n_layers, n_single_qubit_params=3)

    train_circuits = [ansatz(diagram) for diagram in train_diagrams]
    dev_circuits =  [ansatz(diagram) for diagram in dev_diagrams]
    test_circuits = [ansatz(diagram) for diagram in test_diagrams]

    return train_circuits, dev_circuits, test_circuits

def set_model(model_string,checkpoint,logdir=''):
    match model_string:
        case "Numpy":
            if checkpoint:
                    model = NumpyModel.from_checkpoint(logdir+'\model.lt')
            else:
                    model = NumpyModel.from_diagrams(all_circuits, use_jit=True)
        case "Tket":
            backend = AerBackend()
            backend_config = {
                'backend': backend,
                'compilation': backend.default_compilation_pass(2),
                'shots': 8192
            }
            model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
    
    return model

def save_everything(logdir,loss_function,acc_function,a,c,A,model,trainer,test_acc):
    print("Saving everything")
    acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

    fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
    ax_tl.set_title('Training set')
    ax_tr.set_title('Development set')
    ax_bl.set_xlabel('Iterations')
    ax_br.set_xlabel('Iterations')
    ax_bl.set_ylabel('Accuracy')
    ax_tl.set_ylabel('Loss')

    colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    range_ = np.arange(1, trainer.epochs + 1)
    ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
    ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
    ax_tr.plot(range_, trainer.val_costs, color=next(colours))
    ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))
    plt.savefig(logdir+'\plot.png')


    best_model=NumpyModel.from_checkpoint(logdir+'\\best_model.lt')
    best_model_test_acc = acc(best_model(test_circuits), test_labels)
    model=NumpyModel.from_checkpoint(logdir+'\\model.lt')
    test_acc = acc(model(test_circuits), test_labels)

    file_path = f"{logdir}/info_file.txt"
    with open(file_path, 'w') as file:
        # Write the input string to the file
        input_string=f"""Task: Meaning classification
    Classical Embeddings: GloVe 50-d
    Parsing: True
    Rewritign: Remove Cups
    Ansatz: {ansatz_string}
    Layers: {n_layers}
    Map: [N:{map[N]}, S:{map[S]}]
    Model: Numpy
    Backend: None
    Trainer: Quantum Trainer
    Loss function: {loss_function}
    Accuracy function: {acc_function}
    Optimizer: SPSA optimizer
    Epochs: {EPOCHS}
    Batch size: {BATCH_SIZE}
    Seed: {SEED}
    Hyperparams: [a:{a},c:{c},A:{A}]
    Test accuracy: {test_acc}
    Test accuracy best model: {best_model_test_acc}"""
        file.write(input_string)


def main(EPOCHS, SEED, BATCH_SIZE,MODEL):
    # Using the builtin binary cross-entropy error from lambeq
    acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
    bce = BinaryCrossEntropyLoss()
    loss_function="BindaryCrosEntropyLoss"
    acc_function="lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2"

    a=0.05
    c=0.06
    A="0.1*Epochs"
    logdir='runs\Baseline\Epochs_{}--A_{}--N_{}--S_{}--L_{}--Ansatz_{}\Seed_{}'.format(EPOCHS,a,map[N],map[S],n_layers,ansatz_string,SEED)

    trainer = QuantumTrainer(
        model=MODEL,
        loss_function=bce,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': a, 'c': 0.06, 'A':0.01*EPOCHS},
        evaluate_functions={'acc': acc},
        evaluate_on_train=True,
        verbose = 'text',
        seed=SEED,
        from_checkpoint=checkpoint,
        log_dir=logdir
    )

    train_dataset = Dataset(
                train_circuits,
                train_labels,
                batch_size=BATCH_SIZE)

    val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)

    now = datetime.datetime.now()
    t = now.strftime("%Y-%m-%d_%H_%M_%S")
    print(t)

    trainer.fit(train_dataset, val_dataset, log_interval=10)
    test_acc = ''#acc(model(test_circuits), test_labels)

    save_everything(logdir=logdir,loss_function=loss_function,acc_function=acc_function,a=a,c=c,A=A,model=MODEL,trainer=trainer,test_acc=test_acc)


preq_embeddings=load_data()
# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE
map={N:2,S:1}

n_layers=1

alpha="Sim15Ansatz"
beta="FslBase"
gamma="FslSim15"
ansatz_string=beta

print("Turning sentences to circuits")
print(ansatz_string)
print(map)
train_circuits, dev_circuits, test_circuits=create_circuits(map=map,n_layers=n_layers,ansatz_string=ansatz_string,preq_embeddings=preq_embeddings)

all_circuits = train_circuits+dev_circuits+test_circuits

checkpoint=False

print("Setting model")
model=set_model(model_string="Numpy",checkpoint=checkpoint)

seed_arr = [0, 10, 50, 77, 100, 111, 150, 169, 200, 234, 250, 300, 350, 400, 450]
B_sizes = [1250]
epochs_arr = [2000]

for SEED in seed_arr:
    for BATCH_SIZE in B_sizes:
        for EPOCHS in epochs_arr:
            print(EPOCHS, SEED, BATCH_SIZE)
            main(EPOCHS, SEED, BATCH_SIZE,MODEL=model)

now = datetime.datetime.now()
t = now.strftime("%Y-%m-%d_%H_%M_%S")
print(t)

