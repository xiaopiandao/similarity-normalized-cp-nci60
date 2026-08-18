import unittest, numpy as np
from scripts.evaluate_phase3_local_and_screening import screening
class Phase3LocalTests(unittest.TestCase):
 def test_screening_counts(self):
  y=np.array([[6.2,6.1,6.4],[6.1,6.3,np.nan]]); p=y.copy(); lo=np.array([[6.05,6.01,6.1],[6.0,6.05,0.]])
  cell,mol=screening(y,p,lo,'x',np.array([.3,.7])); self.assertEqual(cell['selected'],4); self.assertEqual(mol['selected'],1)
if __name__=='__main__':unittest.main()
