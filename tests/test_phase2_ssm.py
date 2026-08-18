import unittest
import torch
from scripts.run_phase2_ssm import build_model
class Phase2ModelTests(unittest.TestCase):
 def test_model_shapes(self):
  m=build_model(32,9); reg,cls=m(torch.zeros(3,2048),torch.zeros(60,32),torch.zeros(60,dtype=torch.long)); self.assertEqual(tuple(reg.shape),(3,60)); self.assertEqual(tuple(cls.shape),(3,60,6))
if __name__=='__main__': unittest.main()
