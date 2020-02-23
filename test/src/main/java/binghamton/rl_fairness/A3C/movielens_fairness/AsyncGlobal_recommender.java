package binghamton.rl_fairness.A3C.movielens_fairness;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.gradient.Gradient;
import org.jfree.util.Log;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import com.google.common.collect.Iterables;

import binghamton.rl.NeuralNet;
import binghamton.rl.A3C.ActorCriticSeparate;
import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.WeightLearningApplication;
import edu.umd.cs.psl.application.learning.weight.em.HardEM;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.util.GroundKernels;
import edu.umd.cs.psl.application.util.Grounding;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.evaluation.result.FullInferenceResult;
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer;
import edu.umd.cs.psl.groovy.syntax.GenericVariable;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.PersistedAtomManager;
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.reasoner.admm.ADMMReasoner;
import edu.umd.cs.psl.util.database.Queries;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;


public class AsyncGlobal_recommender<NN extends NeuralNet> extends Thread {
	ActorCriticSeparate current;
	final private ConcurrentLinkedQueue<Pair<Gradient[], Integer>> queue;
	final private int maxStep;
	
	private AtomicInteger T = new AtomicInteger(0);
	
	public AsyncGlobal_recommender(ActorCriticSeparate initial, int maxStep) {
		this.current = initial;
		this.maxStep = maxStep;
		queue = new ConcurrentLinkedQueue<>();
	}
	
	public boolean isTrainingComplete() {
		return T.get() >= maxStep;
	}
	
	public void enqueue(Gradient[] gradient, Integer nstep) {
		queue.add(new Pair<>(gradient, nstep));
	}
	
	public ActorCriticSeparate getCurrent() {
		return current;
	}
	
	public AtomicInteger getT() {
		return T;
	}
	
	@Override
	public void run() {
		System.out.println("Global Thread Started!");
		while(!isTrainingComplete()) {
			if(!queue.isEmpty()) {
				Pair<Gradient[], Integer> pair = queue.poll();
				T.addAndGet(pair.getSecond());
				Gradient[] gradient = pair.getFirst();
				synchronized (this) {
					current.applyGradient(gradient, pair.getSecond());
				}
			}
		}
	}
}







