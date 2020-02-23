package binghamton.fairness;

import edu.umd.cs.psl.reasoner.admm.WeightedObjectiveTerm;


public class fairnessConstraintTerm extends fairnessADMMObjectiveTerm implements WeightedObjectiveTerm {

	
	fairnessConstraintTerm(fairnessADMMReasoner reasoner, int[] zIndices, double[] coeffs, double weight) {
		super(reasoner, zIndices);
	}
	
	@Override
	public void setWeight(double weight) {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected void minimize() {
		// TODO Auto-generated method stub
		
	}
	
}
