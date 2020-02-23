package binghamton.fairness;

import edu.umd.cs.psl.application.groundkernelstore.GroundKernelStore;
import edu.umd.cs.psl.model.atom.AtomEvent;
import edu.umd.cs.psl.model.atom.AtomEventFramework;
import edu.umd.cs.psl.model.atom.AtomManager;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.model.parameters.Parameters;



public class fairnessKernel implements Kernel {

	@Override
	public void notifyAtomEvent(AtomEvent event) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int groundAll(AtomManager atomManager, GroundKernelStore gks) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void registerForAtomEvents(AtomEventFramework eventFramework, GroundKernelStore gks) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void unregisterForAtomEvents(AtomEventFramework eventFramework, GroundKernelStore gks) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Parameters getParameters() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setParameters(Parameters para) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Kernel clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		return null;
	}
	
}
