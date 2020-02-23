package binghamton.test.compasExperiment;


import java.text.DecimalFormat;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.umd.cs.psl.application.inference.*;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.util.GroundKernels
import edu.umd.cs.psl.application.util.Grounding
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.application.learning.weight.em.PairedDualLearner;

import edu.umd.cs.psl.config.*;
import edu.umd.cs.psl.core.*;
import edu.umd.cs.psl.core.inference.*;
import edu.umd.cs.psl.database.*;
import edu.umd.cs.psl.database.rdbms.*;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type;
import edu.umd.cs.psl.evaluation.result.*;
import edu.umd.cs.psl.evaluation.statistics.*;
import edu.umd.cs.psl.groovy.*;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.atom.*;
import edu.umd.cs.psl.model.formula.*;
import edu.umd.cs.psl.model.function.*;
import edu.umd.cs.psl.model.kernel.*;
import edu.umd.cs.psl.model.kernel.linearconstraint.GroundLinearConstraint
import edu.umd.cs.psl.model.predicate.*;
import edu.umd.cs.psl.model.term.*;
import edu.umd.cs.psl.model.rule.*;
import edu.umd.cs.psl.model.weight.*;
import edu.umd.cs.psl.reasoner.Reasoner
import edu.umd.cs.psl.reasoner.ReasonerFactory
import edu.umd.cs.psl.reasoner.admm.ADMMReasonerFactory
import edu.umd.cs.psl.reasoner.function.FunctionComparator
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.parameters.Weight

import edu.umd.cs.psl.ui.loading.*;
import edu.umd.cs.psl.util.database.*;

import com.google.common.collect.Iterables;

import edu.umd.cs.psl.util.database.Queries;

import edu.umd.cs.psl.evaluation.resultui.printer.*;

import java.io.*;
import java.util.*;

import groovy.time.*;

import binghamton.util.FoldUtils
import binghamton.util.GroundingWrapper
import binghamton.fairnessInference.compasMPEInference
import binghamton.util.AUCcalculator
import binghamton.util.DataOutputter;


// Config manager
ConfigManager cm = ConfigManager.getManager();
ConfigBundle config = cm.getBundle("compas-model");
Logger log = LoggerFactory.getLogger(this.class);

// Database
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "compas-model")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

def fold = 0;
def dataDir = 'data/compas_5cv/'+fold+ '/test'


def fairnessSignal = true;
def delta = 0.1

PSLModel model = new PSLModel(this, data)

model.add predicate: "user", types: [ArgumentType.UniqueID]
model.add predicate: "recid", types: [ArgumentType.UniqueID]
// Age
model.add predicate: "oldAge", types: [ArgumentType.UniqueID]
// Charge Degree
model.add predicate: "felony", types: [ArgumentType.UniqueID]
// Prison History
model.add predicate: "juvFelHistory", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "juvMisdHistory", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "juvOtherHistory", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "priors", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
//model.add predicate: "charge", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
// Days in Jail
model.add predicate: "longJailDay", types: [ArgumentType.UniqueID]

// COMPAS predicted score
model.add predicate: "compasScore", types: [ArgumentType.UniqueID]

// Sensitive Attributes
model.add predicate: "race", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "sex", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

def closedPredicates = [user, oldAge, felony, juvFelHistory,
	juvMisdHistory, juvOtherHistory, priors, longJailDay,
	compasScore, race, sex] as Set;
def inferredPredicates = [recid] as Set;

def predicateFileMap = [
	((Predicate)user): "user.txt",
	((Predicate)recid): "recidivist.txt",

	((Predicate)oldAge): "oldAge.txt",
	((Predicate)felony): "felony.txt",
	
	((Predicate)juvFelHistory): "juvFelHistory.txt",
	((Predicate)juvMisdHistory): "juvMisdHistory.txt",
	((Predicate)juvOtherHistory): "juvOtherHistory.txt",
	((Predicate)priors): "priors.txt",
//	((Predicate)charge): "chargeNum.txt",
	
	((Predicate)longJailDay): "longJailDay.txt",
	
	((Predicate)compasScore): "compas_score.txt",
	((Predicate)race): "race.txt",
	((Predicate)sex): "sex.txt"]

def predicateSoftTruthMap = [
	((Predicate)user): false,
	((Predicate)recid): true,
	
	((Predicate)oldAge): true,
	((Predicate)felony): true,
	
	((Predicate)juvFelHistory): false,
	((Predicate)juvMisdHistory): false,
	((Predicate)juvOtherHistory): false,
	((Predicate)priors): false,
//	((Predicate)charge): false,
	
	((Predicate)longJailDay): true,
	
	((Predicate)compasScore): true,
	((Predicate)race): false,
	((Predicate)sex): false]


/*
 * Load PSL Model
 */

def initWeight = 5.0

model.add(rule: (oldAge(U)) >> ~recid(U), squared:true, weight: initWeight)
model.add(rule: (user(U) & ~oldAge(U)) >> recid(U), squared: true, weight: initWeight)

model.add(rule: (felony(U)) >> recid(U), squared: true, weight: initWeight)
model.add(rule: (user(U) & ~felony(U)) >> ~recid(U), squared:true, weight: initWeight)

model.add(rule: (juvFelHistory(U, I1)) >> recid(U), squared:true, weight: initWeight)
model.add(rule: (juvMisdHistory(U,I2)) >> recid(U), squared:true, weight: initWeight)
model.add(rule: (juvOtherHistory(U, I3)) >> recid(U), squared:true, weight: initWeight)
model.add(rule: (priors(U, I4)) >> recid(U), squared:true, weight: initWeight)
//model.add(rule: (charge(U, I5)) >> recid(U), squared:true, weight: initWeight)

model.add(rule: (longJailDay(U)) >> recid(U), squared:true, weight: initWeight)
model.add(rule: (user(U) & ~longJailDay(U)) >> ~recid(U), squared:true, weight: initWeight)

model.add(rule: (user(U)) >> ~recid(U), squared:true, weight: initWeight)

if (!fairnessSignal) {
	// Sensitive
	model.add(rule: race(U, 'African-American') >> recid(U), squared:true, weight: initWeight)
}


BufferedReader reader = null;
def file;
if (!fairnessSignal)
	file = '/baisedBaselineModel.txt'
else
	file = '/fairnessBaselineModel.txt'
reader = new BufferedReader(new FileReader('result/compas/fold'+ fold+ file));
String read = null;
List<Double> weights = new ArrayList<Double>();
while ((read = reader.readLine()) != null) {
	def value = Double.valueOf(read)
	weights.add(value);
}
reader.close();


Map<String, Integer> truthMap = new HashMap<>();
String fileName = dataDir+ '/recidivist.txt'
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	int value = Integer.parseInt(splited[1]);
	truthMap.put(splited[0], value);
}


int idx = 0;
for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)) {
	k.setWeight(new PositiveWeight(weights.get(idx)));
	idx++;
}

Partition trainPart = new Partition(0)
Partition truthPart = new Partition(1)
Partition inferenceWritePart = new Partition(3)

def inserter;
for (Predicate p : closedPredicates) {
	println p.getName()
	fileName = predicateFileMap[p];
	inserter = data.getInserter(p, trainPart);
	def fullFilePath = dataDir+ '/'+ fileName;
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

for (Predicate p: inferredPredicates) {
	println p.getName()
	fileName = predicateFileMap[p];
	inserter = data.getInserter(p, truthPart);
	def fullFilePath = dataDir+ '/'+ fileName;
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

void populateDatabase(DataStore data, Database dbToPopulate, Partition populatePartition, Set inferredPredicates){
	Database populationDatabase = data.getDatabase(populatePartition, inferredPredicates);
	DatabasePopulator dbPop = new DatabasePopulator(dbToPopulate);

	for (Predicate p : inferredPredicates){
		dbPop.populateFromDB(populationDatabase, p);
	}
	populationDatabase.close();
}

Database inferredDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
populateDatabase(data, inferredDB, truthPart, inferredPredicates);

Database wlTruthDB = data.getDatabase(truthPart, inferredPredicates);

/*
 * Analysis Sensitive Attributes
 */
Set<String> protectedGroup = new HashSet<>();
int protectedNum = 0;
int unprotectedNum = 0;
int posProtectedNum = 0;
int negProtectedNum = 0;
int posUnprotectedNum = 0;
int negUnprotectedNum = 0;

String raceFile = dataDir+ '/race.txt'
reader = null;
reader = new BufferedReader(new FileReader(raceFile));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	user = splited[0];
	race = splited[1];
	if (race.equals("African-American")) {
		protectedGroup.add(user);
		protectedNum++;
		if (truthMap.get(user) == 1) {
			posProtectedNum++;
		} else {
			negProtectedNum++;
		}
	} else {
		unprotectedNum++;
		if (truthMap.get(user) == 1) {
			posUnprotectedNum++;
		} else {
			negUnprotectedNum++;
		}
	}
}


/*
 * Inference
 */
println "Doing Inference"

def mpe;
if (fairnessSignal)
	mpe = new compasMPEInference(model, inferredDB, config, protectedGroup, protectedNum, unprotectedNum, delta);
else
	mpe = new MPEInference(model, inferredDB, config);
def result = mpe.mpeInference();

def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC, RankingScore.Kendall]

SimpleRankingComparator comparator = new SimpleRankingComparator(inferredDB);
comparator.setBaseline(wlTruthDB);
double[] score = new double[metrics.size()];
for (int r=0; r<metrics.size(); r++) {
	comparator.setRankingScore(metrics[r]);
	score[r] = comparator.compare(recid);
}

double protectedValue = 0;
double unprotectedValue = 0;

double posProtectedValue = 0;
double posUnprotectedValue = 0;
double negProtectedValue = 0;
double negUnprotectedValue = 0;

for (GroundAtom atom : Queries.getAllAtoms(inferredDB, recid)) {
	GroundTerm[] terms = atom.getArguments();
	double value = atom.getValue();
	String user = terms[0].toString()
	if (!protectedGroup.contains(user)) { // Unprotected
		unprotectedValue += value;
		if (truthMap.get(user) == 1) {
			posUnprotectedValue += value;
		} else {
			negUnprotectedValue += value;
		}
	} else { // Protected
		protectedValue += value;
		if (truthMap.get(user) == 1) {
			posProtectedValue += value;
		} else {
			negProtectedValue += value;
		}
	}
}


println model.toString()

println "Area under positive PR curve: "+ score[0]
println "Area under negative PR curve: "+ score[1]
println "Area under ROC curve: "+ score[2]

double p1 = protectedValue / protectedNum;
double p2 = unprotectedValue / unprotectedNum;
println "P1: "+ p1
println "P2: "+ p2
println "RD: "+ (p1-p2)
println "RR: "+ (p1/p2)
println "RC: "+ ((1-p1) / (1-p2))



// Equalized Odds
double posProtectedRate = posProtectedValue / posProtectedNum;
double negProtectedRate = negProtectedValue / negProtectedNum;
double posUnprotectedRate = posUnprotectedValue / posUnprotectedNum;
double negUnprotectedRate = negUnprotectedValue / negUnprotectedNum;

println "pos protected rate: "+ posProtectedRate
println "pos Unprotected rate: "+ posUnprotectedRate
println "neg protected rate: "+ negProtectedRate
println "pos unprotected rate: "+ negUnprotectedRate
println "pos differ: "+ (posProtectedRate- posUnprotectedRate)
println "neg differ: "+ (negProtectedRate- negUnprotectedRate)


List<Double> protectedValueList = new ArrayList<Double>();
List<Double> unprotectedValueList = new ArrayList<Double>();
List<Integer> protectedTruthList = new ArrayList<Integer>();
List<Integer> unprotectedTruthList = new ArrayList<Integer>();
for (GroundAtom atom : Queries.getAllAtoms(inferredDB, recid)) {
	GroundTerm[] terms = atom.getArguments();
	double value = atom.getValue();
	String user = terms[0].toString()
	if (!protectedGroup.contains(user)) { // Unprotected
		unprotectedValueList.add(value);
		unprotectedTruthList.add(truthMap.get(user));
	} else { // Protected
		protectedValueList.add(value);
		protectedTruthList.add(truthMap.get(user));
		
	}
}

double[] protectedValueArr = protectedValueList.toArray();
double[] unprotectedValueArr = unprotectedValueList.toArray();
int[] protectedTruthArr = protectedTruthList.toArray();
int[] unprotectedTruthArr = unprotectedTruthList.toArray();

AUCcalculator calculator = new AUCcalculator();
double protectedAUC = calculator.measure(protectedTruthArr, protectedValueArr);
double unprotectedAUC = calculator.measure(unprotectedTruthArr, unprotectedValueArr);
println "AUC-ROC Proteted: "+ protectedAUC
println "AUC-ROC Unprotected: "+ unprotectedAUC

