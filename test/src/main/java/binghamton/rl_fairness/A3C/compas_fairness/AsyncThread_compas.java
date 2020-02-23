package binghamton.rl_fairness.A3C.compas_fairness;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.Stack;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.log4j.spi.LoggerFactory;
import org.deeplearning4j.nn.gradient.Gradient;
import org.jfree.util.Log;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.google.common.collect.Iterables;

import binghamton.fairnessInference.compasMPEInference;
import binghamton.rl.NeuralNet;
import binghamton.rl.A3C.ActorCriticSeparate;
import binghamton.rl.A3C.IActorCritic;
import binghamton.rl.A3C.MiniTrans;
import binghamton.rl_fairness.A3C.compasPSLModelCreation;
import binghamton.util.AUCcalculator;
import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.WeightLearningApplication;
import edu.umd.cs.psl.application.learning.weight.em.HardEM;
import edu.umd.cs.psl.application.learning.weight.semantic_em.SemanticHardEM;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.semantic.SemanticMaxLikelihoodMPE;
import edu.umd.cs.psl.application.util.GroundKernels;
import edu.umd.cs.psl.application.util.Grounding;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.config.ConfigManager;
import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.database.loading.Inserter;
import edu.umd.cs.psl.evaluation.result.FullInferenceResult;
import edu.umd.cs.psl.evaluation.result.memory.MemoryFullInferenceResult;
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer;
import edu.umd.cs.psl.groovy.syntax.GenericVariable;
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.PersistedAtomManager;
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.model.kernel.linearconstraint.GroundLinearConstraint;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.reasoner.Reasoner;
import edu.umd.cs.psl.reasoner.ReasonerFactory;
import edu.umd.cs.psl.reasoner.admm.ADMMReasoner;
import edu.umd.cs.psl.reasoner.admm.ADMMReasonerFactory;
import edu.umd.cs.psl.reasoner.function.FunctionComparator;
import edu.umd.cs.psl.util.database.Queries;


public class AsyncThread_compas <NN extends NeuralNet> extends Thread {
	Random rdm;
	
	private int threadNumber;
	private int stepCounter = 0;
	private int epochCounter = 0;
	final int nstep;
	final double gamma;
	// Fairness Parameters
	final double delta;
	final int protectedNum;
	final int unprotectedNum;
	final Set<String> protectedGroup;
//	final Map<String, Integer> truthMap;
//	final int[] protectedNegPos;
//	final int[] unprotectedNegPos;
	
	StandardPredicate[] X;
	StandardPredicate[] Y;
	StandardPredicate[] Z;
	StandardPredicate dummyPred;
	StandardPredicate[] allPosPreds;
	StandardPredicate[] allNegPreds;
	final Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap;
	final Set<StandardPredicate> SensitiveAttrs;
	
	// Right Reason
	final Set<String> Recid_Signal;
	final Set<String> Inrecid_Signal;
	
	final int maxRuleLen;
	final int maxRuleNum;
	final int posPredNum;
	final int negPredNum;
	final int ReturnAction;
	
	PSLModel model;
	DataStore data;
	final Database wlTruthDB;
	final Partition trainPart;
	
	int[][] ruleListEmbedding;
	final int inputRow;
	final int inputCol;
	final int outputSize;
	
	private NN current;
	private AsyncGlobal_compas<NN> asyncGlobal;
	
	private ConfigBundle config;
	
	public AsyncThread_compas(AsyncGlobal_compas<NN> asyncGlobal, int threadNum, 
			int maxRuleLen, int maxRuleNum, int nstep, double gamma, double delta) {
		this.threadNumber = threadNum;
		this.asyncGlobal = asyncGlobal;
		this.rdm = new Random();
		
		this.maxRuleLen = maxRuleLen;
		this.maxRuleNum = maxRuleNum;
		this.nstep = nstep;
		this.gamma = gamma;
		this.delta = delta;
		
		compasPSLModelCreation generator = new compasPSLModelCreation(threadNumber);
		this.data = generator.getData();
		this.model = generator.getModel();
		this.config = generator.getConfig();
		this.wlTruthDB = generator.getWlTruthDB();
		this.trainPart = generator.getTrainPart();
		
		this.protectedGroup = generator.getProtectedGroup();
		this.protectedNum = generator.getProtectedNum();
		this.unprotectedNum = generator.getUnprotectedNum();
		
		this.X = generator.getX();
		this.Y = generator.getY();
		this.Z = generator.getZ();
		this.dummyPred = generator.getDummyPred();
		this.SensitiveAttrs = generator.getSensitiveAttributes();
		this.generalPredArgsMap = generator.getGeneralPredArgsMap();
		X = ArrayUtils.removeElement(X, dummyPred);
		for (StandardPredicate p : SensitiveAttrs)
			X = ArrayUtils.removeElement(X, p);
		
		this.Recid_Signal = generator.getRecid_Signal();
		this.Inrecid_Signal = generator.getInrecid_Signal();
		
		StandardPredicate[] negFeatPreds = generator.getNegFeatPreds();
		this.allPosPreds = ArrayUtils.addAll(ArrayUtils.addAll(X, Y), Z);
		this.allNegPreds = ArrayUtils.addAll(ArrayUtils.addAll(negFeatPreds, Y), Z);
		this.posPredNum = allPosPreds.length;
		this.negPredNum = allNegPreds.length;
		
		this.outputSize = posPredNum+ negPredNum+ 1; // // Positive, Negative Predicates, and Return
		this.inputRow = maxRuleNum;
		this.inputCol = outputSize;
		this.ReturnAction = outputSize-1;
		
		this.ruleListEmbedding = new int[inputRow][inputCol];
		
		synchronized (asyncGlobal) {
			current = (NN)asyncGlobal.getCurrent().clone();
		}
	}
	
	@Override
	public void run() {
		try {
			System.out.println("ThreadNum-"+ threadNumber+ " Started!");

			double bestScore = 0;
			int bestEpochIdx = 0;
			String bestRuleList = "";
			
			int nextAccessIdx;
			int[] lastRulePreds = new int[posPredNum];
			int currentAct;
			stepCounter = 0;
			double reward;
			int kernelSize = 0;
			double accumulatedReward;
			
			boolean END_SIGNAL;
			boolean RULE_END_SIGNAL;
			
			boolean TARGET_PENALTY_SIGNAL;
			boolean GRAMMAR_PENALTY_SIGNAL;
			
			int t_start;
			
			/*
			 * Initialization
			 */
			current.reset();
			resetRuleListEmbedding();
			cleanPSLrule();
			nextAccessIdx = 0;
			IntStream.range(0, posPredNum).forEach(r->lastRulePreds[r]=0);
			END_SIGNAL = false;
			RULE_END_SIGNAL = false;
			accumulatedReward = 0;
			
			TARGET_PENALTY_SIGNAL = false; // Check on each Rule
			GRAMMAR_PENALTY_SIGNAL = false; // Check on each Rule
			
			while(!asyncGlobal.isTrainingComplete()) {
				t_start = stepCounter;
				
				synchronized (asyncGlobal) {
					current.copy(asyncGlobal.getCurrent());
			    }
				Stack<MiniTrans> rewards = new Stack<MiniTrans>();
			
				while (!END_SIGNAL && (stepCounter-t_start < nstep)) {
					stepCounter++;
					INDArray observation = processHistory();
					currentAct = nextAction();
					
					/*
					 * State-Action Transition
					 */
					if (!RULE_END_SIGNAL) {
						if (currentAct == ReturnAction) { // Rule Return
							ruleListEmbedding[nextAccessIdx][ReturnAction] = 1;
							RULE_END_SIGNAL = true;
						} else {
							int posPredIdx = getPosPredIndex(currentAct);
							lastRulePreds[posPredIdx] += 1;
							if (lastRulePreds[posPredIdx] == 1) {
								ruleListEmbedding[nextAccessIdx][currentAct] = 1;
							}
							if (IntStream.of(lastRulePreds).sum() >= maxRuleLen) {
								ruleListEmbedding[nextAccessIdx][ReturnAction] = 1;
								RULE_END_SIGNAL = true;
							}
						}
					}
					if (RULE_END_SIGNAL && IntStream.of(lastRulePreds).sum()>0) {
						// Check Target
						if (lastRulePreds[posPredNum-1]==0)
							TARGET_PENALTY_SIGNAL = true;
						// Check Grammar
						if (!TARGET_PENALTY_SIGNAL) {
							GRAMMAR_PENALTY_SIGNAL = !buildNewRule(ruleListEmbedding[nextAccessIdx]);
						}
					}
					
					/*
					 * Assign Reward
					 */
					reward = 0;
					if (TARGET_PENALTY_SIGNAL || GRAMMAR_PENALTY_SIGNAL) {
						END_SIGNAL = true;
					}
					if (RULE_END_SIGNAL) {
						nextAccessIdx++;
						// Check Finishing Building Rule List
						if (!END_SIGNAL) {
							if ((nextAccessIdx == maxRuleNum) || (IntStream.of(lastRulePreds).sum()==0)) {
								END_SIGNAL = true;
							}
						}
					}
					if (END_SIGNAL) {
						kernelSize = 0;
						for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)){
							kernelSize++;
						}
						if (kernelSize!=0) { // && checkContainsNegTargetInSeq()) {
							reward = AUCcost();
						} else {
							reward = 0;
						}
					}
					accumulatedReward += reward;
					
					/*
					 * Reset Signals
					 */
					if (RULE_END_SIGNAL) {
						RULE_END_SIGNAL = false;
						IntStream.range(0, posPredNum).forEach(r->lastRulePreds[r]=0);
						TARGET_PENALTY_SIGNAL = false;
						GRAMMAR_PENALTY_SIGNAL = false;
					}
					
					/*
					 * Stack
					 */
					INDArray valueOutput = current.outputAll(observation)[0];
					rewards.add(new MiniTrans(observation, currentAct, valueOutput, reward));
				}
				
				INDArray observation = processHistory();
				if (END_SIGNAL) {
					rewards.add(new MiniTrans(observation, null, null, 0));
				} else {
					INDArray valueOutput = current.outputAll(observation)[0];
					double value = valueOutput.getDouble(0);
					rewards.add(new MiniTrans(observation, null, null, value));
				}
				
				asyncGlobal.enqueue(calcGradient4NN((ActorCriticSeparate) current, rewards), (stepCounter-t_start));
				rewards.clear();
				
				if (END_SIGNAL) {
					if (accumulatedReward > bestScore) {
						bestScore = accumulatedReward;
						bestEpochIdx = epochCounter;
						bestRuleList = model.toString();
						System.out.println("Thread "+ threadNumber+ " Update Best Score: "+ bestScore+ " at Epoch "+ 
								bestEpochIdx+ ", "+bestRuleList);
					}
					
					if (epochCounter % 100 == 0 || accumulatedReward >0.69) {
						System.out.println("Thread-"+ threadNumber+ " [Epoch: "+ epochCounter+ ", Step: "+ stepCounter+ "]"+ 
								", Reward: "+ accumulatedReward+ ", Size: "+ kernelSize);
						if (kernelSize > 1) {
							System.out.println(model.toString());
						}
					}
					epochCounter++;
					
					current.reset();
					resetRuleListEmbedding();
					cleanPSLrule();
					nextAccessIdx = 0;
					END_SIGNAL = false;
					accumulatedReward = 0;
				}
			}
			System.out.println("Thread "+ threadNumber+ " Best Score: "+ bestScore+ 
					" at Epoch "+ bestEpochIdx+ ", "+bestRuleList);
		}
		catch (Exception e) {
			System.out.println("Thread crashed: "+ e);
		}
	}
	
	public void resetRuleListEmbedding() {
		IntStream.range(0,inputRow).forEach(i -> 
			IntStream.range(0, inputCol).forEach(j-> ruleListEmbedding[i][j]=0));
	}
	
	public void cleanPSLrule() {
		Iterator<Kernel> kernels = model.getKernels().iterator();
		List<Kernel> kernelList = new ArrayList<>();
		for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)){
			kernelList.add(k);
		}
		for (int i=0; i<kernelList.size(); i++) {
			model.removeKernel(kernelList.get(i));
		}
	}
	
	public int getPosPredIndex(int currentAct) {
		if (currentAct==ReturnAction)
			return -1;
		int predIdx = -1;
		if (currentAct < posPredNum) { // Positive Predicate
			return currentAct;
		} else { // Negative Predicate
			int negPredIdx = currentAct- posPredNum;
			StandardPredicate p = allNegPreds[negPredIdx];
			for (int r=0; r<posPredNum; r++) {
				if (allPosPreds[r] == p) {
					predIdx = r;
					break;
				}
			}
		}
		return predIdx;
	}
	
	public boolean checkContainsNegTargetInSeq() {
		int posTargetSignal = 0;
		int targetIdx = posPredNum+ negPredNum-1;
		for (int i=0; i<inputRow; i++) {
			if (ruleListEmbedding[i][targetIdx]==1) {
				posTargetSignal++;
			}
		}
		return posTargetSignal>0 ? true:false;
	}
	
	public Integer nextAction() {
		INDArray observation = processHistory();
		INDArray policyOutput = current.outputAll(observation)[1].reshape(new int[] {outputSize});
		float rVal = rdm.nextFloat();

		for (int i=0; i<policyOutput.length(); i++) {
			if (rVal < policyOutput.getFloat(i)) {
				return i;
			}
			else {
				rVal -= policyOutput.getFloat(i);
			}
		}
        throw new RuntimeException("Output from network is not a probability distribution: " + policyOutput);
	}
	
	public INDArray processHistory() {
//		System.out.println("Process Observation");
		INDArray observation;
		observation = Nd4j.zeros(new int[] {1, inputRow*inputCol});
		for (int r=0; r<inputRow; r++) {
			if (IntStream.of(ruleListEmbedding[r]).sum() == 0)
				break;
			for (int c=0; c<inputCol; c++) {
				if (ruleListEmbedding[r][c]==1)
					observation.putScalar(new int[] {0, r*inputCol+c}, 1.0);
			}
		}
		return observation;
	}
	
	public Gradient[] calcGradient4NN(ActorCriticSeparate iac, Stack<MiniTrans> rewards) {
		MiniTrans minTrans = rewards.pop();
		int size = rewards.size();
		
		int[] nshape = new int[] {size, inputRow*inputCol}; 
		INDArray input = Nd4j.create(nshape);
		INDArray targets = Nd4j.create(size, 1);
		INDArray logSoftmax = Nd4j.zeros(size, outputSize);
		
		double r = minTrans.reward;
		for (int i=size-1; i>=0; i--) {
			minTrans = rewards.pop();
			r = minTrans.reward+ gamma*r;
			 input.get(NDArrayIndex.point(i), NDArrayIndex.all()).assign(minTrans.obs);
			
			// the critic
			targets.putScalar(i, r);
			// the actor
			double expectedV = minTrans.valueOutput.getDouble(0);
			double advantage = r- expectedV;
			logSoftmax.putScalar(i, minTrans.action, advantage);
		}

		Gradient[] gradients = iac.gradient(input, new INDArray[] {targets, logSoftmax});
		return gradients;
	}
	
	
	public double AUCcost() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		final RankingScore[] metrics = new RankingScore[] {RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC};
		final double[] LAMBDA_AUC = new double[] {0, 0, 1.0};
		
		// Define Partitions and Databases
		Partition targetPart = new Partition(threadNumber*10+2);
		Partition inferenceWritePart = new Partition(threadNumber*10+3);
		
		Set<StandardPredicate> closedPredicates = new HashSet<StandardPredicate>(Arrays.asList(X));
		Database wlTrainDB = data.getDatabase(targetPart, closedPredicates, trainPart);
		Database inferenceDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
		
		for (int j=0; j<Y.length; j++) {
			ResultList allGroundings = wlTruthDB.executeQuery(Queries.getQueryForAllAtoms(Y[j]));
			for (int i=0; i<allGroundings.size(); i++) {
				GroundTerm [] grounding = allGroundings.get(i);
				GroundAtom atom = inferenceDB.getAtom(Y[j], grounding);
				if (atom instanceof RandomVariableAtom) {
					wlTrainDB.commit((RandomVariableAtom) atom);
					inferenceDB.commit((RandomVariableAtom) atom);
				}
			}
		}
		
//		// Calculate Semantic Distances
//		double[] dists = calculateSemanticDist();
		
		// Weight Learning
		WeightLearningApplication weightLearner = new MaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, config);
		
//		WeightLearningApplication weightLearner = new SemanticMaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, dists, config);
		
		ArrayList<Integer> numGroundedList = weightLearner.learn();
	    weightLearner.close();
	    wlTrainDB.close();
	    data.deletePartition(targetPart);
	    
	    double loss_grounded = (numGroundedList.stream().mapToInt(Integer::intValue).sum())*1.0/numGroundedList.size();

	    // Do inference
//		MPEInference mpe = new MPEInference(model, inferenceDB, config);
	    compasMPEInference mpe = new compasMPEInference(model, inferenceDB, config,
	    		protectedGroup, protectedNum, unprotectedNum, delta);
	    FullInferenceResult result = mpe.mpeInference();
		mpe.close();
		inferenceDB.close();
		
		Set<StandardPredicate> InferredSet = new HashSet<StandardPredicate>(Arrays.asList(Y)); 
		Database resultDB = data.getDatabase(inferenceWritePart, InferredSet);
		SimpleRankingComparator comparator = new SimpleRankingComparator(resultDB);
		comparator.setBaseline(wlTruthDB);
		double[] score = new double[metrics.length];
		for (int r=0; r<metrics.length; r++) {
			comparator.setRankingScore(metrics[r]);
			score[r] = comparator.compare(Y[0]);
		}
		resultDB.close();
		data.deletePartition(inferenceWritePart);
		
		double reward_auc = 0;
		for (int i=0; i<LAMBDA_AUC.length; i++) {
			reward_auc += score[i]* LAMBDA_AUC[i];
		}
		if (Double.isNaN(reward_auc)) {
			reward_auc = 1e-10;
		}
		double reward = reward_auc; //+ reward_grounded;
		if (loss_grounded == 0) 
			reward = 0;
		return reward;
	}
	
	
	public double[] calculateSemanticDist() { // Distance to Right Reasons
		int kernelSize = 0;
		for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class))
			kernelSize++;
		
		double[] dists = new double[kernelSize];
		final double satisfy_dist = 0.0;
		final double notSatisfy_dist = 1.0;
		
		for (int i=0; i<kernelSize; i++) {			
			boolean EXIST_RECIDSIGNAL = false;
			boolean EXIST_INRECIDSIGNAL = false;
			
			boolean TARGET_POSITIVE_HEAD;
			int target_idx;
			if (ruleListEmbedding[i][posPredNum-1]==1) {
				target_idx = posPredNum-1;
				TARGET_POSITIVE_HEAD = false;
			} else {
				target_idx = posPredNum+negPredNum-1;
				assert ruleListEmbedding[i][target_idx]==1 : "No Target in Rule";
				TARGET_POSITIVE_HEAD = true;
			}
			
			for (int j=0; j<ReturnAction; j++) {
				if (ruleListEmbedding[i][j]==1 && j!=target_idx) {
					if (j < posPredNum) { // Positive 
						String p_name = allPosPreds[j].getName();
						if (Recid_Signal.contains(p_name)) 
							EXIST_RECIDSIGNAL = true;
						else if (Inrecid_Signal.contains(p_name))
							EXIST_INRECIDSIGNAL = true;
					} else { // Negative
						String p_name = "~"+ allNegPreds[j-posPredNum].getName();
						if (Recid_Signal.contains(p_name))
							EXIST_RECIDSIGNAL = true;
						else if (Inrecid_Signal.contains(p_name))
							EXIST_INRECIDSIGNAL = true;
					}
				}
			}
			
			// Calculate Distance to Right Reasons
			if (EXIST_RECIDSIGNAL && EXIST_INRECIDSIGNAL)
				dists[i] = satisfy_dist;
			else if (!EXIST_RECIDSIGNAL && !EXIST_INRECIDSIGNAL)
				dists[i] = satisfy_dist;
			else if (EXIST_RECIDSIGNAL && !EXIST_INRECIDSIGNAL) {
				if (TARGET_POSITIVE_HEAD)
					dists[i] = satisfy_dist;
				else
					dists[i] = notSatisfy_dist;
			}
			else {
				if (TARGET_POSITIVE_HEAD)
					dists[i] = notSatisfy_dist;
				else
					dists[i] = satisfy_dist;
			}
		}
		return dists;
	}
	
	public boolean buildNewRule(int[] ruleEmbedding) {
		FormulaContainer body = null;
		FormulaContainer head = null;
		FormulaContainer rule = null;
		final double initWeight = 5.0;
		List<GenericVariable> argsList;
		Object[] args;
		
		// Add Dummy Predicate to Body
		int[] posPart = new int[posPredNum];
		IntStream.range(0, posPredNum).forEach(r-> posPart[r]=ruleEmbedding[r]);
		if (IntStream.of(posPart).sum()==0) {
			argsList = generalPredArgsMap.get(dummyPred);
			args = argsList.toArray();
			body = (FormulaContainer) model.createFormulaContainer(dummyPred.getName(), args);
		}
		
		// Randomly Choose Head Predicate
		List<Integer> potentialHeadPreds = new ArrayList<Integer>();
		for (int i=0; i<ReturnAction; i++) {
			if (ruleEmbedding[i] == 1) {
				potentialHeadPreds.add(i);
			}
		}
		int headIdx = potentialHeadPreds.get(rdm.nextInt(potentialHeadPreds.size()));
		if (headIdx < posPredNum) { // Negative Head
			StandardPredicate p = allPosPreds[headIdx];
			argsList = generalPredArgsMap.get(p);
			args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
			head = (FormulaContainer) head.bitwiseNegate();
		} else { // Positive Head
			StandardPredicate p = allNegPreds[headIdx-posPredNum];
			argsList = generalPredArgsMap.get(p);
			args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
		}
		
		for (int i=0; i<ReturnAction; i++) {
			if (ruleEmbedding[i]==1 && i!=headIdx) {
				if (i<posPredNum) { // Positive Body Predicate
					StandardPredicate p = allPosPreds[i];
					argsList= generalPredArgsMap.get(p);
					args = new Object[argsList.size()];
					args = argsList.toArray(args);
					if (body==null)
						body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					else {
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						body = (FormulaContainer) body.and(f_tmp);
					}
				} else { // Negative Body Predicate
					StandardPredicate p = allNegPreds[i-posPredNum];
					argsList= generalPredArgsMap.get(p);
					args = new Object[argsList.size()];
					args = argsList.toArray(args);
					if (body==null) {
						body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						body = (FormulaContainer) body.bitwiseNegate();
					} else {
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						f_tmp = (FormulaContainer) f_tmp.bitwiseNegate();
						body = (FormulaContainer) body.and(f_tmp);
					}
				}
			}
		}
		
		if (body == null) {
			rule = head;
		} else {
			rule = (FormulaContainer) body.rightShift(head);
		}
		
		boolean succeed;
		Map<String, Object> argsMap = new HashMap<String, Object>();
		argsMap.put("rule", rule);
		argsMap.put("squared", true);
		argsMap.put("weight", initWeight);
		try {
			model.add(argsMap);
			succeed = true;
		} catch (Exception e) {
			Log.error(e);
			succeed = false;
		}
		return succeed;
	}
	
}



