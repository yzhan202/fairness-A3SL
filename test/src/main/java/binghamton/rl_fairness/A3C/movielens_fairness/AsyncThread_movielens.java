package binghamton.rl_fairness.A3C.movielens_fairness;

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

import binghamton.fairnessInference.recommenderMPEInference;
import binghamton.rl.NeuralNet;
import binghamton.rl.A3C.ActorCriticSeparate;
import binghamton.rl.A3C.IActorCritic;
import binghamton.rl.A3C.MiniTrans;
import binghamton.rl_fairness.A3C.recommenderPSLModelCreation;
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

import binghamton.util.Graph;


public class AsyncThread_movielens<NN extends NeuralNet> extends Thread {
	Random rdm;
	
	private int threadNumber;
	private int stepCounter = 0;
	private int epochCounter = 0;
	final int nstep;
	final double gamma;
	// Fairness Parameters
	final double sigma;
	final Set<Integer> protectedGroup;
	final Map<Integer, Integer> protectedItemNum;
	final Map<Integer, Integer> unprotectedItemNum;
	final int protectedNum;
	final int unprotectedNum;
	final Map<Integer, Double> avgProtectedItemScore;
	final Map<Integer, Double> avgUnprotectedItemScore;
	
	final Map<String, Double> truthMap;
	final Set<Integer> ItemSet;
	
	StandardPredicate[] X;
	StandardPredicate[] Y;
	StandardPredicate[] Z;
	StandardPredicate[] dummyPreds;
	StandardPredicate[] allPosPreds;
	StandardPredicate[] allNegPreds;
	final Map<StandardPredicate, List<List<GenericVariable>>> generalPredArgsMap;
	final Set<StandardPredicate> SensitiveAttribute;
	Map<StandardPredicate, Integer> posPredIdxMap;
	Map<StandardPredicate, Integer> negPredIdxMap;
	
	/*
	 * Right Reason
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
	
	private NN current;
	private AsyncGlobal_recommender<NN> asyncGlobal;
	
	final double MIN_VAL = 1e-5;
	private ConfigBundle config;
	
	public AsyncThread_movielens(AsyncGlobal_recommender<NN> asyncGlobal, int threadNum, 
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
		
		recommenderPSLModelCreation generator = new recommenderPSLModelCreation(threadNum);
		this.data = generator.getData();
		this.model = generator.getModel();
		this.config = generator.getConfig();
		this.wlTruthDB = generator.getWlTruthDB();
		this.trainPart = generator.getTrainPart();
		
		this.protectedGroup = generator.getProtectedGroup();
		this.protectedItemNum = generator.getProtectedItemNum();
		this.unprotectedItemNum = generator.getUnprotectedItemNum();
		this.protectedNum = generator.getProtectedNum();
		this.unprotectedNum = generator.getUnprotectedNum();
		this.avgProtectedItemScore = generator.getAveProtectedItemScore();
		this.avgUnprotectedItemScore = generator.getAveUnprotectedItemScore();
		this.truthMap = generator.getTruthMap();
		this.ItemSet = generator.getItemSet();

		this.X = generator.getX();
		this.Y = generator.getY();
		this.Z = generator.getZ();
		this.dummyPreds = new StandardPredicate[] {(StandardPredicate)PredicateFactory.getFactory().getPredicate("user"),
				(StandardPredicate)PredicateFactory.getFactory().getPredicate("item"),
				(StandardPredicate)PredicateFactory.getFactory().getPredicate("reviews"),
				(StandardPredicate)PredicateFactory.getFactory().getPredicate("userLink"),
				(StandardPredicate)PredicateFactory.getFactory().getPredicate("itemLink")};
		this.SensitiveAttribute = generator.getSensitiveAttribute();
		this.generalPredArgsMap = generator.getGeneralPredArgsMap();
		
		for (StandardPredicate p: dummyPreds)
			X = ArrayUtils.removeElement(X, p);
		for (StandardPredicate p : SensitiveAttribute)
			X = ArrayUtils.removeElement(X, p);
		StandardPredicate[] negFeatPreds = generator.getNegFeatPreds();
		
		initializeActionSpace(negFeatPreds);
		this.posPredNum = allPosPreds.length;
		this.negPredNum = allNegPreds.length;
		
		// Right ReasonITEMLINK
		this.Positive_Signal = generator.getPositive_Signal();
		this.Negative_Signal = generator.getNegative_Signal();
		
		this.outputSize = posPredNum+ negPredNum+ 1;
		this.inputRow = maxRuleNum;
		this.inputCol = outputSize;
		this.ReturnAction = outputSize-1;
		
		this.ruleListEmbedding = new int[inputRow][inputCol];
		
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
		for (int i=0; i<shape.length; i++) {
			nshape[1] *= shape[i];
		}
		nshape[2] = length;
		return nshape;
	}
	
	@Override
	public void run() { // throws ClassNotFoundException, IllegalAccessException, InstantiationException {
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
							//synchronized (asyncGlobal) {
							reward = AUCcost();
							//}
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
					
					if (epochCounter % 20 == 0 || accumulatedReward >= 0.85) {
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
		// final RankingScore[] metrics = new RankingScore[] {RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC};
		// final double[] LAMBDA_AUC = new double[] {0, 0, 1.0};
//		System.out.println(model.toString());
		Set<StandardPredicate> InferredSet = new HashSet<StandardPredicate>(Arrays.asList(Y));
		
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
		
		// Calculate Semantic Distances
//		double[] dists = calculateSemanticDist();
		
		// Do Weight Learning
//		System.out.println("Do weight Learning");
		ArrayList<Integer> numGroundedList;
		WeightLearningApplication weightLearner = null;
		weightLearner = new MaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, config);
//		weightLearner = new SemanticMaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, dists, config);
		
		numGroundedList = weightLearner.learn();
		weightLearner.close();
		wlTrainDB.close();
		data.deletePartition(targetPart);
		
//	    double loss_grounded = (numGroundedList.stream().mapToInt(Integer::intValue).sum())*1.0/numGroundedList.size();

	    // Dp Fairness Inference
//	    System.out.println("Do Inference");
	    recommenderMPEInference mpe = new recommenderMPEInference(model, inferenceDB, config,
	    		protectedGroup, protectedNum, unprotectedNum, sigma);
	    FullInferenceResult result = mpe.mpeInference();
		mpe.close();
		inferenceDB.close();
		
		Database resultDB = data.getDatabase(inferenceWritePart,  InferredSet);
		
		/*
		 * calculate Value Unfairness
		 */
//		System.out.println("Calculate Metrics");
		Map<String, Double> predictionMap = new HashMap<>();
		double unfairness_value = 0;
		Map<Integer, Double> avgProtectedItemPred = new HashMap<>();
		Map<Integer, Double> avgUnprotectedItemPred = new HashMap<>();
		for (GroundAtom atom : Queries.getAllAtoms(resultDB, Y[0])) {
			GroundTerm[] terms = atom.getArguments();
			double value = atom.getValue();
			int user = Integer.parseInt(terms[0].toString());
			int item = Integer.parseInt(terms[1].toString());
			predictionMap.put((terms[0].toString()+','+terms[1].toString()), value);
			if (protectedGroup.contains(user)) { // Protected User
				if (avgProtectedItemPred.containsKey(item)) {
					double tmp_r = avgProtectedItemPred.get(item);
					avgProtectedItemPred.put(item, tmp_r+value);
				} else { // Unprotected User
					avgProtectedItemPred.put(item, value);
				}
			} else { // Unprotected User
				if (avgUnprotectedItemPred.containsKey(item)) {
					double tmp_r = avgUnprotectedItemPred.get(item);
					avgUnprotectedItemPred.put(item, tmp_r+value);
				} else {
					avgUnprotectedItemPred.put(item, value);
				}
			}
		}
		
		int n_item = ItemSet.size(); // 0;
		for (int v : ItemSet) {
			double predProtect_value = 0;
			double protected_value = 0;
			if (avgProtectedItemScore.containsKey(v)) {
				predProtect_value = avgProtectedItemPred.get(v) / (1.0*(protectedItemNum.get(v)));
				protected_value = avgProtectedItemScore.get(v);
			}
			
			double predUnprotect_value = 0;
			double unprotected_value = 0;
			if (avgUnprotectedItemScore.containsKey(v)) {
				predUnprotect_value = avgUnprotectedItemPred.get(v) / (1.0*(unprotectedItemNum.get(v)));
				unprotected_value = avgUnprotectedItemScore.get(v);
			}
			unfairness_value += Math.abs(Math.max(0, predProtect_value- protected_value) - Math.max(0, predUnprotect_value- unprotected_value));
		}
		unfairness_value /= n_item*1.0;
		double reward_unfairnessValue = -0.5 * unfairness_value; // 0.1; 0.5; 0.0; 
		
		double error = 0;
		for (Map.Entry<String, Double> entry: predictionMap.entrySet()) {
			String key = entry.getKey();
			double pred = entry.getValue();
			double truth = truthMap.get(key);
			error +=  (pred-truth)*(pred-truth);
		}
		error /= truthMap.size()*1.0;
		
		resultDB.close();
	    data.deletePartition(inferenceWritePart);
	    //System.out.println("Err: "+ error + ", Over-estimation: "+ unfairness_value);
	    double reward = Math.pow(10.0, -error)+ reward_unfairnessValue;

//	    if (loss_grounded == 0)
//	    	reward = 0;

	    return reward;	  
	}

	public FormulaContainer addDummyPred(int[] ruleEmbedding) {
		FormulaContainer body = null;
		
		Set<String> linkSet = new HashSet<>();
		Set<String> nodeSet = new HashSet<>();
		for (int i=0 ; i<ReturnAction; i++) {
			if (ruleEmbedding[i] == 1) {
				StandardPredicate p;
				int deltaIdx;
				if (i < posPredNum) {
					p = allPosPreds[i];
					deltaIdx = i- posPredIdxMap.get(p);
				}
				else {
					p = allNegPreds[i-posPredNum];
					deltaIdx = i-posPredNum- negPredIdxMap.get(p);
				}
				List<GenericVariable> arg = generalPredArgsMap.get(p).get(deltaIdx);
				if (arg.size() == 1)
					nodeSet.add(arg.get(0).toString());
				else {
					linkSet.add(arg.get(0).toString()+','+ arg.get(1).toString());
					nodeSet.add(arg.get(0).toString());
					nodeSet.add(arg.get(1).toString());
				}
			}
		}
		
		if (nodeSet.contains("U") && nodeSet.contains("U2")) {
			if (!linkSet.contains("U,U2")) {
				List<GenericVariable> arg = generalPredArgsMap.get(dummyPreds[3]).get(0);
				body = (FormulaContainer) model.createFormulaContainer("userLink", arg.toArray());  
			
			}
		}
		
		if (nodeSet.contains("I") && nodeSet.contains("I2")) {
			if (!linkSet.contains("I,I2")) {
				List<GenericVariable> arg = generalPredArgsMap.get(dummyPreds[4]).get(0);
				if (body == null) {
					body = (FormulaContainer) model.createFormulaContainer("itemLink", arg.toArray());  
				} else {				 
					FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer("itemLink", arg.toArray()); 
					body = (FormulaContainer) body.and(f_tmp);
				}			
			}
		}

	/*	Graph g1 = new Graph(nodeSet.size());
		Map<String, Integer> nodeMap = new HashMap<String, Integer>();
		Map<Integer, String> nodeIdxMap = new HashMap<Integer, String>();
		int idx = 0;
		for (String node: nodeSet) {
			nodeMap.put(node, idx);
			nodeIdxMap.put(idx, node);
			idx += 1;
		}
		for (String link : linkSet) {
			String[] splited = link.split(",");
			g1.addEdge(nodeMap.get(splited[0]), nodeMap.get(splited[1]));
			g1.addEdge(nodeMap.get(splited[1]), nodeMap.get(splited[0]));
		}
		Boolean[] visited = g1.isCG();
		for (int i=0; i<visited.length; i++) {
			if (!visited[i]) {
				if (nodeIdxMap.get(i).contains("U")) { // User Signal
					List<GenericVariable> arg = generalPredArgsMap.get(dummyPreds[3]).get(0);
					if (body == null) {
						body = (FormulaContainer) model.createFormulaContainer("userLink", arg.toArray());  
					} else {				 
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer("userLink", arg.toArray()); 
						body = (FormulaContainer) body.and(f_tmp);
					}
				} else { // Item Signal
					List<GenericVariable> arg = generalPredArgsMap.get(dummyPreds[4]).get(0);
					if (body == null) {
						body = (FormulaContainer) model.createFormulaContainer("itemLink", arg.toArray());  
					} else {				 
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer("itemLink", arg.toArray()); 
						body = (FormulaContainer) body.and(f_tmp);
					}
				}
			}
		}*/
		StandardPredicate rating = (StandardPredicate)PredicateFactory.getFactory().getPredicate("rating");
		for (int i=negPredIdxMap.get(rating)+posPredNum; i<ReturnAction; i++) {
			if (ruleEmbedding[i] == 1) {
				int deltaIdx = i-posPredNum- negPredIdxMap.get(rating);
				List<GenericVariable> arg = generalPredArgsMap.get(rating).get(deltaIdx);
				if (body == null) {
					body = (FormulaContainer) model.createFormulaContainer("REVIEWS", arg.toArray());
				} else {
					FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer("REVIEWS", arg.toArray());
					body = (FormulaContainer) body.and(f_tmp);
				}
			}
		}
		
		return body;
	}
	
	public boolean buildNewRule(int[] ruleEmbedding) {
		FormulaContainer body = null;
		FormulaContainer head = null;
		FormulaContainer rule = null;
		final double initWeight = 1.0;
		
		/*
		 *  Add Dummy Predicates to Body
		 */
		body = addDummyPred(ruleEmbedding);
		
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


