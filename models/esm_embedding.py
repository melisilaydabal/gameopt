import sys
import esm
import esm.pretrained
import pickle
import torch
import random
import time
import numpy as np

# Note: new ESM library gives error on pretrained. Hence, I use the old version of esm in python 3.11
# from transformers import AutoTokenizer, AutoModelForMaskedLM

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ESMEmbedding():

    def __init__(self, name = 'facebook/esm1v_t33_650M_UR90S_5', preloaded = None):
        if name == 'esm-1v':
            self.model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_5()
            self.batch_converter = alphabet.get_batch_converter()
            self.alphabet = alphabet

        elif name == 'esm-2':
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = alphabet.get_batch_converter()
            self.alphabet = alphabet
        else:
            raise AssertionError("Not Implemented.")
        self.m = 1280

        if preloaded is not None:
            self.dict = pickle.load(open(preloaded, 'rb'))
        else:
            self.dict = {}

    def dump_embeddings(self, name):
        pickle.dump(self.dict, open(name,"wb"))

    def load_embeddings(self,name):
        self.dict = pickle.load(open(name,"rb"))

    def get_m(self):
        return self.m

    def decode(self, embeddings, max_length=512, num_return_sequences=1):
        decoded_sequences = []
        for embedding in embeddings:
            # Generate token IDs from embeddings
            input_ids = self._embedding_to_input_ids(embedding, max_length, num_return_sequences)
            for ids in input_ids:
                # Convert token IDs to amino acid sequences
                sequence = self._token_ids_to_sequence(ids)
                decoded_sequences.append(sequence)
        return decoded_sequences

    def _embedding_to_input_ids(self, embedding, max_length, num_return_sequences):
        # Use the model to generate token IDs from embeddings
        with torch.no_grad():
            embedding = torch.tensor(embedding, device=device).unsqueeze(0)
            outputs = self.model.generate(
                inputs_embeds=embedding,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                num_beams=5,  # Beam search for generating better sequences
                early_stopping=True
            )
        return outputs

    def _token_ids_to_sequence(self, token_ids):
        # Map token IDs to amino acid sequences
        sequence = ''.join([self.alphabet.get_tok(int(tok)) for tok in token_ids])
        return sequence

    def embed_parallel(self, seq, device):
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.model = self.model.eval()
        print(self.model)
        print(next(self.model.parameters()).device)
        # turn data into this format: data = [("protein1", "MYLYQKIKN"), ("protein2", "MNAKYD")]
        data = [(str(pos), sequence) for pos, sequence in enumerate(seq)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        # Determine the chunk size for each GPU
        chunk_size = batch_tokens.size(0) // num_gpus
        # Split the batch_tokens into equal chunks for each GPU
        batch_tokens_per_gpu = [batch_tokens[i * chunk_size:(i + 1) * chunk_size].to(f'cuda:{i}') for i in
                                range(num_gpus)]

        print('Batch tokens per gpu= {}'.format(np.shape(batch_tokens_per_gpu[0])))

        # Extract per-residue embeddings
        with torch.no_grad():
            # Perform forward pass on each GPU and collect the outputs
            results = [self.model(chunk, repr_layers=[33], return_contacts=True) for chunk in
                       batch_tokens_per_gpu]
            # Concatenate the outputs from all GPUs
            results = torch.cat(results, dim=0)

        token_representations = results["representations"][33]
        # Compute the mean along dimension 1
        token_representations = token_representations.mean(dim=1)
        return token_representations


    def embed(self, seq, device):
        # Based on:        """https://github.com/facebookresearch/esm/blob/dfa524df54f91ef45b3919a00aaa9c33f3356085/README.md#quick-start-"""
        self.model = self.model.eval().to(device)
        # turn data into this format: data = [("protein1", "MYLYQKIKN"), ("protein2", "MNAKYD")]
        data = [(str(pos), sequence) for pos, sequence in enumerate(seq)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        # Get the data type of the model's parameters
        model_precision = self.model.parameters().__next__().dtype
        # Check if the model's precision is float16 (half-precision)
        if model_precision == torch.float16:
            # Convert the input data to float16 before passing it to the model
            batch_tokens = batch_tokens.half()

        # Extract per-residue embeddings
        with torch.no_grad():
            # results = self.model(batch_tokens.cuda(), repr_layers=[33])
            # results = self.model(batch_tokens.cuda(), repr_layers=[33], return_contacts = True)
            results = self.model(batch_tokens.to(device), repr_layers=[33], return_contacts = True)

        token_representations = results["representations"][33]
        # Compute the mean along dimension 1
        token_representations = token_representations.mean(dim=1)

        return token_representations


    def embed_iter(self, seq, device):
        self.model = self.model.eval().to(device)
        out = []
        keys = self.dict.keys()
        for s in seq:
            if s in keys:
                out.append(self.dict[s])
            else:
                batch_labels, batch_strs, batch_tokens = self.batch_converter([("dummy",s)])
                with torch.no_grad():
                    results = self.model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
                    token_representations = results["representations"][33]
                    z = token_representations[0,:,:].mean(0)
                self.dict[s] = z
                out.append(z)
        return torch.vstack(out)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# Main script
if __name__ == '__main__':

    if len(sys.argv) != 5:
        sys.exit("usage: python esm_embedding.py <esm_embedding_model_name> <dataset_size> <dir_dataset> <cuda>\n"
                 "possible ESM Embedding models are: [esm-1v, esm-2]")
    else:
        model_name = sys.argv[1]
        dataset_size = int(sys.argv[2])
        dir_dataset = sys.argv[3]
        cuda_no = sys.argv[4]

    cuda = cuda_no
    if torch.cuda.is_available():
        dev = "cuda:" + cuda
    else:
        dev = "cpu"
    device = torch.device(dev)

    dir_models = './pre-trained_models/'

    # Set the seed for reproducibility
    random.seed(42)  # Set your desired seed value

    with open(dir_models + 'esm_embedding_{}.pkl'.format(model_name), 'rb') as file:
        esm_embedding_model = pickle.load(file)
    print('Using ESM model {}...'.format(model_name))


    dataset_name = 'gfp'
    with open(dir_dataset + f'/{dataset_name}_dict.pkl', 'rb') as file:
        train_data = pickle.load(file)
    if dataset_size == 0:
        dataset_size = len(train_data)

    print('Computing embeddings for train dataset...')
    # Start the timer
    start_time = time.time()

    sequence_list = list(train_data.keys())
    batch_size = 50
    num_batches = len(sequence_list) // batch_size + 1
    print(f'Number of batches to process: {num_batches}')

    train_x_tensors = []
    train_y_tensors = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_sequence = sequence_list[start_idx:end_idx]

        if len(batch_sequence) > 0:
            with torch.no_grad():
                batch_embeddings = esm_embedding_model.embed(batch_sequence, device)
            train_x_tensors.append(batch_embeddings)
            del batch_embeddings
            train_y_tensors.append([train_data[key]['fitness'] for key in batch_sequence])
            print(f'Embedding for batch {i} is generated')

    train_x_tensor = torch.cat(train_x_tensors, dim=0)
    train_y_tensor = torch.tensor([item for sublist in train_y_tensors for item in sublist])

    print('Saving embeddings and dictionaries...')
    with open(dir_dataset + f'/{dataset_name}_x_tensor.pkl', 'wb') as file:
        pickle.dump(train_x_tensor, file)
    with open(dir_dataset + f'/{dataset_name}_y_tensor.pkl', 'wb') as file:
        pickle.dump(train_y_tensor, file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


