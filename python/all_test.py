import unittest


class AllTest(unittest.TestCase):  # pylint: disable=R0904
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_add_subject(self) -> None:
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
