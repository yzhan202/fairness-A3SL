/*
 * This file is part of the PSL software.
 * Copyright 2011-2015 University of Maryland
 * Copyright 2013-2015 The Regents of the University of California
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

/**
 * Infers the most-probable explanation (MPE) state of the
 * {@link RandomVariableAtom RandomVariableAtoms} persisted in a {@link Database},
 * according to a {@link Model}, given the Database's {@link ObservedAtom ObservedAtoms}.
 * <p>
 * The set of RandomVariableAtoms is those persisted in the Database when {@link #mpeInference()}
 * is called. This set must contain all RandomVariableAtoms the Model might access.
 * ({@link DatabasePopulator} can help with this.)
 * 
 * @author Stephen Bach <bach@cs.umd.edu>
 */
public class FairnessMPEInference implements ModelApplication {
	
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
	private Map<String, String> paperAuthorMap;
	private Set<String> protectedGroup;
	private int protectedNum;
	private int unprotectedNum;
	private double sigma;
	
	public FairnessMPEInference(Model model, Database db, ConfigBundle config,
			Map<String, String> paperAuthorMap, Set<String> protectedGroup, 
			int protectedNum, int unprotectedNum, double sigma) 
			throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		this.model = model;
		this.db = db;
		this.config = config;
		
		this.paperAuthorMap = paperAuthorMap;
		this.protectedGroup = protectedGroup;
		this.protectedNum = protectedNum;
		this.unprotectedNum = unprotectedNum;
		
		initialize();
	}
	
	
	private void initialize() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		reasoner = ((ReasonerFactory) config.getFactory(REASONER_KEY, REASONER_DEFAULT)).getReasoner(config);
		atomManager = new PersistedAtomManager(db);
		
//		log.info("Grounding out model.");
		Grounding.groundAll(model, atomManager, reasoner);
		
		/*
		 * Add Fairness Constraints
		 */
		StandardPredicate positiveSummary = (StandardPredicate)PredicateFactory.getFactory().getPredicate("positiveSummary");
		/*
		 *  Inequality Constraint 1: RD <= sigma
		 */
		List<GroundAtom> atomList = new ArrayList<GroundAtom>();
		List<Double> coeffList = new ArrayList<Double>();
		for (GroundAtom atom : Queries.getAllAtoms(db, positiveSummary)) {
			GroundTerm[] terms = atom.getArguments();
			// check if a student affiliates to high-rank institute
			String authorName = paperAuthorMap.get(terms[0].toString());
			atomList.add(atom);
			if (!protectedGroup.contains(authorName)) { // Unprotected Group
				coeffList.add((double)-protectedNum);
			} else { // Protected Group
				coeffList.add((double)unprotectedNum);
			}
		}
		GroundAtom[] rd1Atoms = new GroundAtom[atomList.size()];
		IntStream.range(0, atomList.size()).forEach(r-> rd1Atoms[r]=atomList.get(r));
		double[] rd1Coeffs = coeffList.stream().mapToDouble(Double::doubleValue).toArray();
		GroundLinearConstraint rd1_GLC = new GroundLinearConstraint(rd1Atoms, rd1Coeffs, 
				FunctionComparator.LargerThan, -sigma*protectedNum*unprotectedNum);
		reasoner.addGroundKernel(rd1_GLC);
		
		/*
		 * Inequality Constraint 2: RD >= -sigma
		 */
		GroundAtom[] rd2Atoms = new GroundAtom[atomList.size()];
		IntStream.range(0, atomList.size()).forEach(r-> rd2Atoms[r]=atomList.get(r));
		double[] rd2Coeffs = coeffList.stream().mapToDouble(Double::doubleValue).toArray();
		GroundLinearConstraint rd2_GLC = new GroundLinearConstraint(rd2Atoms, rd2Coeffs, 
				FunctionComparator.SmallerThan, sigma*protectedNum*unprotectedNum);
		reasoner.addGroundKernel(rd2_GLC);
		atomList.clear();
		coeffList.clear();
		
		/*
		 * 3 RR <= 1+sigma
		 */
		for (GroundAtom atom : Queries.getAllAtoms(db, positiveSummary)) {
			GroundTerm[] terms = atom.getArguments();
			// check if a student affiliates to high-rank institute
			String authorName = paperAuthorMap.get(terms[0].toString());
			atomList.add(atom);
			if (!protectedGroup.contains(authorName)) { // Unprotected Group
				coeffList.add((double)-protectedNum*(1+sigma));
			} else { // Protected Group
				coeffList.add((double)unprotectedNum);
			}
		}
		GroundAtom[] rr1Atoms = new GroundAtom[atomList.size()];
		IntStream.range(0, atomList.size()).forEach(r-> rr1Atoms[r]=atomList.get(r));
		double[] rr1Coeffs = coeffList.stream().mapToDouble(Double::doubleValue).toArray();
		GroundLinearConstraint rr1_GLC = new GroundLinearConstraint(rr1Atoms, rr1Coeffs,
				FunctionComparator.LargerThan, -sigma*protectedNum*unprotectedNum);
		reasoner.addGroundKernel(rr1_GLC);
		
		/*
		 * 5: RC <= 1+sigma
		 */
		GroundAtom[] rc1Atoms = new GroundAtom[atomList.size()];
		IntStream.range(0, atomList.size()).forEach(r-> rc1Atoms[r]=atomList.get(r));
		double[] rc1Coeffs = coeffList.stream().mapToDouble(Double::doubleValue).toArray();
		GroundLinearConstraint rc1_GLC = new GroundLinearConstraint(rc1Atoms, rc1Coeffs,
				FunctionComparator.SmallerThan, 0);
		reasoner.addGroundKernel(rc1_GLC);
		atomList.clear();
		coeffList.clear();
		
		/*
		 * 4 RR >= 1-sigma
		 */
		for (GroundAtom atom : Queries.getAllAtoms(db, positiveSummary)) {
			GroundTerm[] terms = atom.getArguments();
			// check if a student affiliates to high-rank institute
			String authorName = paperAuthorMap.get(terms[0].toString());
			atomList.add(atom);
			if (!protectedGroup.contains(authorName)) { // Unprotected Group
				coeffList.add((double)protectedNum*(sigma-1));
			} else { // Protected Group
				coeffList.add((double)unprotectedNum);
			}
		}
		GroundAtom[] rr2Atoms = new GroundAtom[atomList.size()];
		IntStream.range(0, atomList.size()).forEach(r-> rr2Atoms[r]=atomList.get(r));
		double[] rr2Coeffs = coeffList.stream().mapToDouble(Double::doubleValue).toArray();
		GroundLinearConstraint rr2_GLC = new GroundLinearConstraint(rr2Atoms, rr2Coeffs, 
				FunctionComparator.SmallerThan, sigma*protectedNum*unprotectedNum);
		reasoner.addGroundKernel(rr2_GLC);
	
		/*
		 * 6: RC >= 1-sigma
		 */
//		GroundAtom[] rc2Atoms = (GroundAtom[]) atomList.toArray();
		GroundAtom[] rc2Atoms = new GroundAtom[atomList.size()];
		IntStream.range(0, atomList.size()).forEach(r-> rc2Atoms[r]=atomList.get(r));
		double[] rc2Coeffs = coeffList.stream().mapToDouble(Double::doubleValue).toArray();
		GroundLinearConstraint rc2_GLC = new GroundLinearConstraint(rc2Atoms, rc2Coeffs, 
				FunctionComparator.LargerThan, 0);
		reasoner.addGroundKernel(rc2_GLC);
		atomList.clear();
		coeffList.clear();
	}
	
	/**
	 * Minimizes the total weighted incompatibility of the {@link GroundAtom GroundAtoms}
	 * in the Database according to the Model and commits the updated truth
	 * values back to the Database.
	 * <p>
	 * The {@link RandomVariableAtom RandomVariableAtoms} to be inferred are those
	 * persisted in the Database when this method is called. All RandomVariableAtoms
	 * which the Model might access must be persisted in the Database.
	 * 
	 * @return inference results
	 * @see DatabasePopulator
	 */
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
