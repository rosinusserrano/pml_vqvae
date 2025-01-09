@torch.no_grad()
    def sample(
        self,
        class_idx_list: torch.Tensor,
        probabilistic_sampling_prob: float = 1,
    ):

        shape = (len(class_idx_list), 1, *self.input_shape)

        # Create empty image
        imgs = torch.zeros(shape, dtype=torch.float32).to(DEVICE)

        # Generation loop
        for h in range(self.input_shape[0]):
            for w in range(self.input_shape[1]):
                print(h, w)
                preds = self.forward(imgs, class_idx_list)
                if random() < probabilistic_sampling_prob:
                    probs = F.softmax(preds, dim=1)[:, :, h, w]
                    tmp = torch.multinomial(probs, num_samples=1)
                else:
                    tmp = torch.argmax(preds, dim=1)[:, None, h, w]
                imgs[:, :, h, w] = tmp

        return imgs.cpu()