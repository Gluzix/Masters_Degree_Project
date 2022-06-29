from numpy import zeros
from numpy import ones
from numpy.random import randint
from matplotlib import pyplot


class Pix2PixTrainer:
    def __init__(self, discriminator, generator, gan, dataset):
        self.discriminator = discriminator
        self.generator = generator
        self.gan = gan
        self.dataset = dataset

    def train(self, n_epochs=400, n_batch=2):
        n_patch = self.discriminator.output_shape[1]
        trainA, trainB = self.dataset
        bat_per_epo = int(len(trainA) / n_batch)
        n_steps = bat_per_epo * n_epochs
        for i in range(n_steps):
            [X_realA, X_realB], y_real = self.generate_real_samples(self.dataset, n_batch, n_patch)
            X_fakeB, y_fake = self.generate_fake_samples(self.generator, X_realA, n_patch)
            d_loss1 = self.discriminator.train_on_batch([X_realA, X_realB], y_real)
            d_loss2 = self.discriminator.train_on_batch([X_realA, X_fakeB], y_fake)
            g_loss, _, _ = self.gan.train_on_batch(X_realA, [y_real, X_realB])
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
            if (i+1) % (bat_per_epo * 10) == 0:
                self.summarize_performance(i)

    def summarize_performance(self, step, n_samples=3):
        [X_realA, X_realB], _ = self.generate_real_samples(self.dataset, n_samples, 1)
        X_fakeB, _ = self.generate_fake_samples(self.generator, X_realA, 1)
        X_realA = (X_realA + 1) / 2.0
        X_realB = (X_realB + 1) / 2.0
        X_fakeB = (X_fakeB + 1) / 2.0
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(X_realA[i])
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(X_fakeB[i])
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
            pyplot.axis('off')
            pyplot.imshow(X_realB[i])
        filename1 = 'plot_%06d.png' % (step+1)
        pyplot.savefig(filename1)
        pyplot.close()
        filename2 = 'model_%06d.h5' % (step+1)
        self.generator.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))

    @staticmethod
    def generate_fake_samples(generator, samples, patch_shape):
        X = generator.predict(samples)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    @staticmethod
    def generate_real_samples(dataset, n_samples, patch_shape):
        trainA, trainB = dataset
        ix = randint(0, trainA.shape[0], n_samples)
        X1, X2 = trainA[ix], trainB[ix]
        y = ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y

    @staticmethod
    def generate_real_samples_array(dataset):
        trainA, trainB = dataset
        X1, X2 = trainA[:], trainB[:]
        return [X1, X2]
