package binghamton.fairnessInference;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.umd.cs.psl.application.ModelApplication;
import edu.umd.cs.psl.application.util.GroundKernels;
import edu.umd.cs.psl.application.util.Grounding;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.config.ConfigManager;
import edu.umd.cs.psl.config.Factory;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator;
import edu.umd.cs.psl.evaluation.result.FullInferenceResult;
import edu.umd.cs.psl.evaluation.result.memory.MemoryFullInferenceResult;
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.ObservedAtom;
import edu.umd.cs.psl.model.atom.PersistedAtomManager;
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.kernel.linearconstraint.GroundLinearConstraint;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.reasoner.Reasoner;
import edu.umd.cs.psl.reasoner.ReasonerFactory;
import edu.umd.cs.psl.reasoner.admm.ADMMReasonerFactory;
import edu.umd.cs.psl.reasoner.function.FunctionComparator;
import edu.umd.cs.psl.util.database.Queries;



public class recommenderMPEInference implements ModelApplication {
	private static final Logger log = LoggerFactory.getLogger(FairnessMPEInference.class);
	
	/**
	 * Prefix of property keys used by this class.
	 * 
	 * @see ConfigManager
	 */
	public static final String CONFIG_PREFIX = "fairnessmpeinference";
	
	/**
	 * Key for {@link Factory} or String property.
	 * <p>
	 * Should be set to a {@link ReasonerFactory} or the fully qualified
	 * name of one. Will be used to instantiate a {@link Reasoner}.
	 */
	public static final String REASONER_KEY = CONFIG_PREFIX + ".reasoner";
	/**
	 * Default value for REASONER_KEY.
	 * <p>
	 * Value is instance of {@link ADMMReasonerFactory}. 
	 */
	public static final ReasonerFactory REASONER_DEFAULT = new ADMMReasonerFactory();
	
	private Model model;
	private Database db;
	private ConfigBundle config;
	private Reasoner reasoner;
	private PersistedAtomManager atomManager;
	
	// Fairness Parameters
	private Set<Integer> protectedGroup;
	// private Map<Integer, Integer> protectedItemNum;
	// private Map<Integer, Integer> unprotectedItemNum;
	private int protectedNum;
	private int unprotectedNum;
	float[] avgProtectedItemScore;
	float[] avgUnprotectedItemScore;
	
	private double sigma;
	
	public recommenderMPEInference(Model model, Database db, ConfigBundle config, 
			Set<Integer> protectedGroup, int protectedNum, int unprotectedNum, 
			double sigma) throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		
		this.model = model;
		this.db = db;
		this.config = config;
		
		this.protectedGroup = protectedGroup;
		// this.protectedItemNum = protectedItemNum;
		// this.unprotectedItemNum = unprotectedItemNum;
		this.protectedNum = protectedNum;
		this.unprotectedNum = unprotectedNum;
		
		initialize();
	}
	
	private void initialize() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		reasoner = ((ReasonerFactory) config.getFactory(REASONER_KEY, REASONER_DEFAULT)).getReasoner(config);
		atomManager = new PersistedAtomManager(db);
		
		Grounding.groundAll(model, atomManager, reasoner);
		
		/*
		 * Add Fairness Constraints
		 */
		StandardPredicate rating = (StandardPredicate)PredicateFactory.getFactory().getPredicate("rating");
		/*
		 * Non-Parity Unfairness 1: U_par <= sigma
		 */
		List<GroundAtom> atomList = new ArrayList<GroundAtom>();
		List<Double> coeffList = new ArrayList<Double>();
		for (GroundAtom atom: Queries.getAllAtoms(db, rating)) {
			GroundTerm[] terms = atom.getArguments();
			int user = Integer.parseInt(terms[0].toString());
			atomList.add(atom);
			if (protectedGroup.contains(user)) { // Protected User
				coeffList.add((double)unprotectedNum);
			} else { // Unprotected User
				coeffList.add((double)-protectedNum);
			}
		}
		GroundAtom[] par1Atoms = new GroundAtom[atomList.size()];
		IntStream.range(0, atomList.size()).forEach(r->par1Atoms[r]=atomList.get(r));
		double[] par1Coeffs = coeffList.stream().mapToDouble(Double::doubleValue).toArray();
		GroundLinearConstraint par1_GLC = new GroundLinearConstraint(par1Atoms, par1Coeffs, 
				FunctionComparator.SmallerThan, sigma*protectedNum*unprotectedNum);
		reasoner.addGroundKernel(par1_GLC);
		
		/*
		 * Non-Parity Unfairness 2: U_par >= -sigma
		 */
		GroundAtom[] par2Atoms = new GroundAtom[atomList.size()];
		IntStream.range(0, atomList.size()).forEach(r->par2Atoms[r]=atomList.get(r));
		double[] par2Coeffs = coeffList.stream().mapToDouble(Double::doubleValue).toArray();
		GroundLinearConstraint par2_GLC = new GroundLinearConstraint(par2Atoms, par2Coeffs, 
				FunctionComparator.LargerThan, -sigma*protectedNum*unprotectedNum);
		reasoner.addGroundKernel(par2_GLC);
		
		atomList.clear();
		coeffList.clear();
	}
	
	
	public FullInferenceResult mpeInference() {
		reasoner.changedGroundKernelWeights();
		
//		log.info("Beginning inference.");
		reasoner.optimize();
//		log.info("Inference complete. Writing results to Database.");
		
		/* Commits the RandomVariableAtoms back to the Database */
		int count = 0;
		for (RandomVariableAtom atom : atomManager.getPersistedRVAtoms()) {
			// Yue Debug
			double tmp = Double.valueOf(String.format("%.10f", atom.getValue()));
			atom.getVariable().setValue(tmp);
			atom.commitToDB();
			count++;
		}
		
		double incompatibility = GroundKernels.getTotalWeightedIncompatibility(reasoner.getCompatibilityKernels());
		double infeasibility = GroundKernels.getInfeasibilityNorm(reasoner.getConstraintKernels());
		int size = reasoner.size();
		return new MemoryFullInferenceResult(incompatibility, infeasibility, count, size);
	}
	
	public Reasoner getReasoner() {
		return reasoner;
	}

	@Override
	public void close() {
		reasoner.close();
		model=null;
		db = null;
		config = null;
	}
}










