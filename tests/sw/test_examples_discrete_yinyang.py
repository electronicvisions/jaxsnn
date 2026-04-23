import unittest
from jaxsnn.examples.discrete import yinyang


class YinYangExampleTest(unittest.TestCase):
    """ Tests the YinYang example implementation """

    def test_training(self) -> None:
        """ Run YinYang training and inference """
        # run training
        train_args = [
            "--seed=0",
            "--testset-size=2944",
            "--trainset-size=4992",
            "--tau-mem=0.01",
            "--tau-syn=0.005",
            "--v_th=0.6",
            "--dt=0.0005",
            "--hidden-size=120",
            "--epochs=10",
            "--batch-size=64",
            "--lr=0.0005",
            "--lr-decay=0.98",
            "--expected-spikes=0.8"
        ]

        accuracy = yinyang.main(yinyang.get_parser().parse_args(train_args))
        self.assertGreater(
            accuracy, 0.9, "Accuracy should be greater than 90%")


if __name__ == "__main__":
    unittest.main()
