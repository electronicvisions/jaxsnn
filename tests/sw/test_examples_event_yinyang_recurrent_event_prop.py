import unittest
from jaxsnn.examples.event import yinyang_recurrent_event_prop


class YinYangRecurrentEventPropExampleTest(unittest.TestCase):
    """ Tests the YinYang example implementation """

    def test_training(self) -> None:
        """ Run YinYang training and inference """
        # run training
        train_args = [
            "--seed=0",
            "--testset-size=2944",
            "--trainset-size=4992",
            "--t-late=1e-2",
            "--correct-target-time=4.5e-3",
            "--wrong-target-time=5.5e-3",
            "--hidden-size=100",
            "--tau-mem=1e-02",
            "--tau-syn=5e-03",
            "--threshold=1.0",
            "--n-spikes-output=53",
            "--epochs=10",
            "--batch-size=64",
            "--lr=0.005",
            "--lr-decay=0.99",
        ]

        accuracy = yinyang_recurrent_event_prop.main(
            yinyang_recurrent_event_prop.get_parser().parse_args(train_args))
        self.assertGreater(
            accuracy, 0.90, "Accuracy should be greater than 90%")


if __name__ == "__main__":
    unittest.main()
