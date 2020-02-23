package binghamton.rl_fairness.A3C.generator_fairness;

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

import binghamton.fairnessInference.FairnessMPEInference;
import binghamton.rl.NeuralNet;
import binghamton.rl.A3C.ActorCriticSeparate;
import binghamton.rl.A3C.IActorCritic;
import binghamton.rl.A3C.MiniTrans;
import binghamton.rl_fairness.A3C.fairnessPSLModelCreation;
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


public class AsyncThread_fairness<NN extends NeuralNet> extends Thread {
	Random rdm;
	
	private int threadNumber;
	private int stepCounter = 0;
	private int epochCounter = 0;
	final int nstep;
	final double gamma;
	// Fairness Parameter
	final double sigma;
	final int protectedNum;
	final int unprotectedNum;
	final Set<String> protectedGroup;
	final Map<String, String> paperAuthorMap;
	
	final Map<String, Integer> truthMap;
	final int[] protectedNegPos;
	final int[] unprotectedNegPos;
	
	StandardPredicate[] X;
	StandardPredicate[] Y;
	StandardPredicate[] Z;
	StandardPredicate[] dummyPreds;
	StandardPredicate[] allPosPreds;
	StandardPredicate[] allNegPreds;
	final Map<StandardPredicate, List<List<GenericVariable>>> generalPredArgsMap;
//	final Map<StandardPredicate, List<List<Object>>> specificPredArgsMap;
	final Set<StandardPredicate> SensitiveAttribute;
	Map<StandardPredicate, Integer> posPredIdxMap;
	Map<StandardPredicate, Integer> negPredIdxMap;
	
	/*
	 * Right Reasons
	 */
	final Set<String> Positive_Signal;
	final Set<String> Negative_Signal;
	
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
	
	final double MIN_VAL = 1e-5;
	final double LAMBDA_ruleLen = 0.0;//0.5;
	final double LAMBDA_ruleNum = 0.0;//0.5;
	final double LAMBDA_coverage = 0.0;//0.2;
	final double LAMBDA_semantic = 0.0;
	
	private NN current;
	private AsyncGlobal_fairness<NN> asyncGlobal;
	
	private ConfigBundle config;
	
	// // Debug
	// double[] standardY;
	
	public AsyncThread_fairness(AsyncGlobal_fairness<NN> asyncGlobal, int threadNum, 
			int maxRuleLen, int maxRuleNum, int nstep, double gamma, double sigma) {
		this.threadNumber = threadNum;
		this.asyncGlobal = asyncGlobal;
		this.rdm = new Random();
		
		this.maxRuleLen = maxRuleLen;
		this.maxRuleNum = maxRuleNum;
		this.nstep = nstep;
		this.gamma = gamma;
		// Fairness
		this.sigma = sigma;
		
		fairnessPSLModelCreation generator = new fairnessPSLModelCreation(threadNumber);
		this.data = generator.getData();
		this.model = generator.getModel();
		this.config = generator.getConfig();
		this.wlTruthDB = generator.getWlTruthDB();
		this.trainPart = generator.getTrainPart();
		
		this.paperAuthorMap = generator.getPaperAuthorMap();
		this.protectedGroup = generator.getProtectedGroup();
		this.protectedNum = generator.getProtectedNum();
		this.unprotectedNum = generator.getUnprotectedNum();
		this.truthMap = generator.getTruthMap();
		this.protectedNegPos = generator.getProtectedNegPos();
		this.unprotectedNegPos = generator.getUnprotectedNegPos();
		
		X = generator.getX();
		Y = generator.getY();
		Z = generator.getZ();
		dummyPreds = new StandardPredicate[] {(StandardPredicate)PredicateFactory.getFactory().getPredicate("author"),
				(StandardPredicate)PredicateFactory.getFactory().getPredicate("paper"),
				(StandardPredicate)PredicateFactory.getFactory().getPredicate("reviewer")};
		SensitiveAttribute = generator.getSensitiveAttribute();
		this.generalPredArgsMap = generator.getGeneralPredArgsMap();

		for (StandardPredicate p: dummyPreds)
			X = ArrayUtils.removeElement(X, p);
		for (StandardPredicate p : SensitiveAttribute)
			X = ArrayUtils.removeElement(X, p);
		StandardPredicate[] negFeatPreds = generator.getNegFeatPreds();
		
		initializeActionSpace(negFeatPreds);
		this.posPredNum = allPosPreds.length;
		this.negPredNum = allNegPreds.length;
		
		// Right Reason
		this.Positive_Signal = generator.getPositive_Signal();
		this.Negative_Signal = generator.getNegative_Signal();
		
		outputSize = posPredNum+ negPredNum+ 1; // Positive, Negative Predicates, and Return
		
		inputRow = maxRuleNum;
		inputCol = outputSize;
		this.ReturnAction = outputSize- 1;
		
		ruleListEmbedding = new int[inputRow][inputCol];
		
		synchronized (asyncGlobal) {
			current = (NN)asyncGlobal.getCurrent().clone();
		}
	}
	
	void initializeActionSpace(StandardPredicate[] negFeatPreds) {
		posPredIdxMap = new HashMap<StandardPredicate, Integer>();
		List<StandardPredicate> allPosPredList = new ArrayList<StandardPredicate>();
		StandardPredicate[] tmpPosPreds = ArrayUtils.addAll(ArrayUtils.addAll(X, Y), Z);
		for (StandardPredicate p: tmpPosPreds) {
			List<List<GenericVariable>> args = generalPredArgsMap.get(p);
			posPredIdxMap.put(p, allPosPredList.size());
			for (int i=0; i<args.size(); i++) 
				allPosPredList.add(p);
		}
		
		negPredIdxMap = new HashMap<>();
		List<StandardPredicate> allNegPredList = new ArrayList<StandardPredicate>();
		StandardPredicate[] tmpNegPreds = ArrayUtils.addAll(ArrayUtils.addAll(negFeatPreds, Y), Z);
		for (StandardPredicate p: tmpNegPreds) {
			List<List<GenericVariable>> args = generalPredArgsMap.get(p);
			negPredIdxMap.put(p, allNegPredList.size());
			for (int i=0; i<args.size(); i++)
				allNegPredList.add(p);
		}
		
		allPosPreds = new StandardPredicate[allPosPredList.size()];
		allNegPreds = new StandardPredicate[allNegPredList.size()];
		IntStream.range(0, allPosPredList.size()).forEach(r->allPosPreds[r]=allPosPredList.get(r));
		IntStream.range(0, allNegPredList.size()).forEach(r->allNegPreds[r]=allNegPredList.get(r));
	}
	
	public static int[] makeShape(int batch, int[] shape, int length) {
        int[] nshape = new int[3];
        nshape[0] = batch;
        nshape[1] = 1;
        for (int i = 0; i < shape.length; i++) {
            nshape[1] *= shape[i];
        }
        nshape[2] = length;
        return nshape;
    }
	
	@Override
	public void run() { //throws ClassNotFoundException, IllegalAccessException, InstantiationException {
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
				synchronized (asyncGlobal) { // Update Local neural Nets 
					current.copy(asyncGlobal.getCurrent());
			    }
				Stack<MiniTrans> rewards = new Stack<MiniTrans>();
				while(!END_SIGNAL && (stepCounter-t_start < nstep)) {
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
						TARGET_PENALTY_SIGNAL = true;
						for (int r=0; r<Y.length; r++) {
							int startIdx = posPredIdxMap.get(Y[r]);
							int size = generalPredArgsMap.get(Y[r]).size();
							for (int s=0; s<size; s++) {
								if (lastRulePreds[startIdx+s]==1)
									TARGET_PENALTY_SIGNAL = false;
							}
						}
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
						for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)) {
							kernelSize++;
						}
						if (kernelSize!=0) { // && checkContainsPosNegTargetInSeq()) {
							synchronized (asyncGlobal) {
								reward = AUCcost();
							}
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
					 *  Stack
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
					
					if (epochCounter % 100 == 0 || accumulatedReward >= 0.84) {
						System.out.println("Thread-"+ threadNumber+ " [Epoch: "+ epochCounter+ ", Step: "+ stepCounter+ "]"+ 
								", Reward: "+ accumulatedReward+ ", Size: "+ kernelSize);
						if (kernelSize>1 && accumulatedReward>0) {
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
			int deltaIdx = negPredIdx-negPredIdxMap.get(p);
			if (posPredIdxMap.containsKey(p) ) {
				predIdx = posPredIdxMap.get(p)+ deltaIdx;
			}
		}
		return predIdx;
	}	
	
//	boolean checkContainsPosNegTargetInSeq() {
//		int[] PosSignal = new int[Y.length];
//		int[] NegSignal = new int[Y.length];
//		IntStream.range(0, Y.length).forEach(r-> PosSignal[r]=0);
//		IntStream.range(0, Y.length).forEach(r-> NegSignal[r]=0);
//		for (int i=0; i<inputRow; i++) {
//			if (IntStream.of(ruleListEmbedding[i]).sum()==0)
//				break;
//			for (int j=0; j<Y.length; j++) { 
//				if (ruleListEmbedding[i][posPredNum-1-j]==1) // Positive Target
//					PosSignal[j] = 1;
//				if (ruleListEmbedding[i][posPredNum+negPredNum-1-j]==1) // Negative Target
//					NegSignal[j] = 1;
//			}
//		}
//		if (IntStream.of(PosSignal).sum()==Y.length && IntStream.of(NegSignal).sum()==Y.length)
//			return true;
//		else
//			return false;
//	}

	public Integer nextAction() {
		INDArray observation = processHistory();
		INDArray policyOutput = current.outputAll(observation)[1].reshape(new int[] {outputSize});
		float rVal = rdm.nextFloat();

		for (int i=0; i<policyOutput.length(); i++) {
			if (rVal < policyOutput.getFloat(i)) {
//				System.out.println("Epoch: "+epochCounter+", Choose Action: "+ i+ ", Policy Output: "+ policyOutput);
				return i;
			}
			else {
				rVal -= policyOutput.getFloat(i);
			}
		}
        throw new RuntimeException("Output from network is not a probability distribution: " + policyOutput);
	}
	
	public INDArray processHistory() {
		INDArray observation = Nd4j.zeros(new int[] {1, inputRow*inputCol});
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
		
//		int[] shape = new int[] {inputRow, inputCol};
		int[] nshape = new int[] {size, inputRow*inputCol}; //makeShape(size, shape);
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
		
		Set<StandardPredicate> InferredSet = new HashSet<StandardPredicate>(Arrays.asList(Y));
	    int targetPredIdx = -1;
	    for (int i=0; i<Y.length; i++) {
	    	if (Y[i].getName().equals("POSITIVESUMMARY")) 
	    		targetPredIdx=i;
	    }
		
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
//		for (int i=0; i<Y.length; i++) {
//			for (GroundAtom atom : Queries.getAllAtoms(inferenceDB, Y[i])) {
//				atom.getVariable().setValue(0.0);
//			}
//		}
		
		// Calculate Semantic Distances
		double[] dists = calculateSemanticDist();
		
		// Do Weight Learning
		ArrayList<Integer> numGroundedList;
		WeightLearningApplication weightLearner = null;
//		weightLearner = new MaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, config);
		weightLearner = new SemanticMaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, dists, config);

		numGroundedList = weightLearner.learn();
		weightLearner.close();

		wlTrainDB.close();
		data.deletePartition(targetPart);
		
	    double loss_grounded = (numGroundedList.stream().mapToInt(Integer::intValue).sum())*1.0/numGroundedList.size();
	    
	    // Do Inference
	    FairnessMPEInference mpe = new FairnessMPEInference(model, inferenceDB, config,
				paperAuthorMap, protectedGroup, protectedNum, unprotectedNum, sigma);	
//	    MPEInference mpe = new MPEInference(model, inferenceDB, config);
		FullInferenceResult result = mpe.mpeInference();
		mpe.close();
		inferenceDB.close();
		
	    Database resultDB = data.getDatabase(inferenceWritePart, InferredSet);
	    SimpleRankingComparator comparator = new SimpleRankingComparator(resultDB);
	    comparator.setBaseline(wlTruthDB);
	    double[] score = new double[metrics.length];
	    for (int r=0; r<metrics.length; r++) {
	    	comparator.setRankingScore(metrics[r]);
	    	score[r] = comparator.compare(Y[targetPredIdx]);
	    }
	    
	    double reward_auc = score[2];
//	    for (int i=0; i<LAMBDA_AUC.length; i++) {
//	    	reward_auc += score[i]* LAMBDA_AUC[i];
//	    }
		
	    if (Double.isNaN(reward_auc))
	    	reward_auc = MIN_VAL;
	    
	    /*
	     * calculate Equalized Odds Difference
	     */
	    double posProtectedValue = 0;
	    double negProtectedValue = 0;
	    double posUnprotectedValue = 0;
	    double negUnprotectedValue = 0;
	    for (GroundAtom atom : Queries.getAllAtoms(resultDB, Y[targetPredIdx])) {
	    	GroundTerm[] terms = atom.getArguments();
	    	double value = atom.getValue();
	    	String paperName = terms[0].toString();
	    	String authorName = paperAuthorMap.get(paperName);
	    	if (!protectedGroup.contains(authorName)) { // Unprotected
	    		if (truthMap.get(paperName) == 1)
	    			posUnprotectedValue += value;
	    		else
	    			negUnprotectedValue += value;
	    	} else { // Protected
	    		if (truthMap.get(paperName) == 1) 
	    			posProtectedValue += value;
	    		else
	    			negProtectedValue += value;	
	    	}
	    }
	    double posProtectedRate = posProtectedValue / protectedNegPos[1];
	    double negProtectedRate = negProtectedValue / protectedNegPos[0];
	    double posUnprotectedRate = posUnprotectedValue / unprotectedNegPos[1];
	    double negUnprotectedRate = negUnprotectedValue / unprotectedNegPos[0];
	    double pos_diff = Math.abs(posProtectedRate-posUnprotectedRate);
	    double neg_diff = Math.abs(negProtectedRate-negUnprotectedRate);
	    double reward_odds = -0.1*(pos_diff+ neg_diff); // 0.1, 0.5, 0.0
	    
//	    System.out.println("AUC Reward: "+ reward_auc+ ", ["+ pos_diff+ ", "+ neg_diff+ "]");   
	    
	    resultDB.close();
	    data.deletePartition(inferenceWritePart);
	    
	    if (loss_grounded == 0)
	    	reward_auc = 0;
	    double reward = reward_auc; //+ reward_odds;

	    return reward;	    
	}
	
	
	double[] calculateSemanticDist() { // Distance to Right Reasons
		int kernelSize = 0;
		for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class))
			kernelSize++;
		
		double[] dists = new double[kernelSize];
		final double satisfy_dist = 0.0;
		final double notSatisfy_dist = 1.0;
		
		for (int i=0; i<kernelSize; i++) {
			int EXIST_POSITIVE_SIGNAL = 0;
			int EXIST_NEGATIVE_SIGNAL = 0;
			
			for (int j=0; j<ReturnAction; j++) {
				if (ruleListEmbedding[i][j]==1) {
					if (j < posPredNum) { // Positive
						String p_name = allPosPreds[j].getName();
						if (Positive_Signal.contains(p_name))
							EXIST_POSITIVE_SIGNAL += 1;
						else if (Negative_Signal.contains(p_name))
							EXIST_NEGATIVE_SIGNAL += 1;
					} else { // Negative
						String p_name = "~"+ allNegPreds[j-posPredNum].getName();
						if (Positive_Signal.contains(p_name))
							EXIST_POSITIVE_SIGNAL += 1;
						else if (Negative_Signal.contains(p_name))
							EXIST_NEGATIVE_SIGNAL += 1;
					}
				}
			}
			
			// Calculate Distance to Right Reasons
			if (EXIST_POSITIVE_SIGNAL>0 && EXIST_NEGATIVE_SIGNAL>0)
				dists[i] = satisfy_dist;
			else if (EXIST_POSITIVE_SIGNAL==1 && EXIST_NEGATIVE_SIGNAL==0)
				dists[i] = satisfy_dist;
			else if (EXIST_NEGATIVE_SIGNAL==1 && EXIST_POSITIVE_SIGNAL==0)
				dists[i] = satisfy_dist;
			else
				dists[i] = notSatisfy_dist;
		}
		return dists;
	}
	
	public boolean buildNewRule(int[] ruleEmbedding) {
		FormulaContainer body = null;
		FormulaContainer head = null;
		FormulaContainer rule = null;
		final double initWeight = 5.0;
		
		/*
		 *  Add Dummy Predicates to Body
		 */
		Set<GenericVariable> containsAuthorSignal = new HashSet<>(); // A
		Set<GenericVariable> containsPaperSignal = new HashSet<>(); // P
		Set<List<GenericVariable>> containsRPsignal = new HashSet<>();
		for (int i=posPredNum; i<ReturnAction; i++) { // Negative Predicates
			if (ruleEmbedding[i] == 1) { 
				StandardPredicate p = allNegPreds[i-posPredNum];
				int deltaIdx = i-posPredNum- negPredIdxMap.get(p);
				List<GenericVariable> arg = generalPredArgsMap.get(p).get(deltaIdx);
				if (arg.size()==1) {
					String argName = arg.get(0).getName();
					if (argName.contains("A"))
						containsAuthorSignal.add(arg.get(0));
					else if (argName.contains("P"))
						containsPaperSignal.add(arg.get(0));
				} else { // (arg.size() == 2)
					String name1 = arg.get(0).getName();
					String name2 = arg.get(1).getName();
					if (name1.contains("R") && name2.contains("P")) {
						StandardPredicate pred= (StandardPredicate)PredicateFactory.getFactory().getPredicate("reviews");
						int tmpIdx = posPredIdxMap.get(pred)+ deltaIdx;
						if (ruleEmbedding[tmpIdx] != 1)
							containsRPsignal.add(arg);
					}
				}
			}
		}
		// itemDummyPred = {author, paper, reviewer}
		for (GenericVariable gv: containsAuthorSignal) {
			if (body == null) {
				body = (FormulaContainer) model.createFormulaContainer(dummyPreds[0].getName(), new Object[] {gv});
			} else {
				FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(dummyPreds[0].getName(), new Object[] {gv});
				body = (FormulaContainer) body.and(f_tmp);
			}
		}
		for (GenericVariable gv: containsPaperSignal) {
			if (body == null) {
				body = (FormulaContainer) model.createFormulaContainer(dummyPreds[1].getName(), new Object[] {gv});
			} else {
				FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(dummyPreds[1].getName(), new Object[] {gv});
				body = (FormulaContainer) body.and(f_tmp);
			}
		}
		for (List<GenericVariable> l: containsRPsignal) {
			if (body == null) {
				body = (FormulaContainer) model.createFormulaContainer("REVIEWS", l.toArray());
			} else {
				FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer("REVIEWS", l.toArray());
				body = (FormulaContainer) body.and(f_tmp);
			}
		}
		
		/*
		 * Randomly Choose Head Predicate
		 */
		List<Integer> potentialHeadPreds = new ArrayList<>();
		for (int i =0; i<ReturnAction; i++) {
			if (ruleEmbedding[i]==1) {
				potentialHeadPreds.add(i);
			}
		}
		int headIdx = potentialHeadPreds.get(rdm.nextInt(potentialHeadPreds.size()));
		if (headIdx < posPredNum) { // Negative Head
			StandardPredicate p = allPosPreds[headIdx];
			int deltaIdx = headIdx- posPredIdxMap.get(p);
			List<GenericVariable> argList = generalPredArgsMap.get(p).get(deltaIdx);
			Object[] args = argList.toArray();
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
			head = (FormulaContainer) head.bitwiseNegate();
		} else { // Positive Head
			StandardPredicate p = allNegPreds[headIdx-posPredNum];
			int deltaIdx = headIdx-posPredNum- negPredIdxMap.get(p);
			List<GenericVariable> argList = generalPredArgsMap.get(p).get(deltaIdx);
			Object[] args = argList.toArray();
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
		}
		
		/*
		 * Build Body
		 */
		for (int i=0; i<ReturnAction; i++) {
			if (ruleEmbedding[i]==1 && i!=headIdx) {
				if (i<posPredNum) { // Positive Body Predicate
					StandardPredicate p = allPosPreds[i];
					int deltaIdx = i- posPredIdxMap.get(p);
					List<GenericVariable> argList = generalPredArgsMap.get(p).get(deltaIdx);
					Object[] args = argList.toArray();
					if (body == null) {
						body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					} else {
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						body = (FormulaContainer) body.and(f_tmp);
					}
				} else { // Negative Body Predicate
					StandardPredicate p = allNegPreds[i-posPredNum];
					int deltaIdx = i-posPredNum- negPredIdxMap.get(p);
					List<GenericVariable> argList = generalPredArgsMap.get(p).get(deltaIdx);
					Object[] args = argList.toArray();
					if (body == null) {
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
		
		if (body==null) {
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







