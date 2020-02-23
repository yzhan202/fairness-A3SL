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
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.*;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type;
import edu.umd.cs.psl.evaluation.result.*;
import edu.umd.cs.psl.evaluation.statistics.*;
import edu.umd.cs.psl.groovy.*;
import edu.umd.cs.psl.groovy.PSLModel;

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

def fold = 4;
def dataDir = 'data/compas_5cv/'+fold+ '/test' // test; training

PSLModel model = new PSLModel(this, data)

model.add predicate: "recid", types: [ArgumentType.UniqueID]

// Sensitive Attributes
model.add predicate: "race", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "sex", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//def closedPredicates = [race, sex] as Set;
def inferredPredicates = [recid] as Set;

def predicateFileMap = [
	((Predicate)recid): "recidivist.txt",
	((Predicate)race): "race.txt",
	((Predicate)sex): "sex.txt"]

def predicateSoftTruthMap = [
	((Predicate)recid): true,
	((Predicate)race): false,
	((Predicate)sex): false]


Map<String, Integer> truthMap = new HashMap<>();
String fileName = dataDir+ '/recidivist.txt'
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	int value = Integer.parseInt(splited[1]);
	truthMap.put(splited[0], value);
}



Partition truthPart = new Partition(1)
Partition inferenceWritePart = new Partition(3)

def inserter;

for (Predicate p: inferredPredicates) {
	fileName = 'compas_score.txt';
	inserter = data.getInserter(p, inferenceWritePart);
	def fullFilePath = dataDir+ '/'+ fileName;
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

for (Predicate p: inferredPredicates) {
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

Database inferredDB = data.getDatabase(inferenceWritePart, inferredPredicates);

Database wlTruthDB = data.getDatabase(truthPart, inferredPredicates);

/*
 * Analysis Sensitive Attributes
 */
Set<String> protectedGroup = new HashSet<>();
int protectedNum = 0;
int unprotectedNum = 0;

int posProtectedNum = 0;
int posUnprotectedNum = 0;
int negProtectedNum = 0;
int negUnprotectedNum = 0;

String raceFile = dataDir+ '/race.txt'
BufferedReader reader = null;
reader = new BufferedReader(new FileReader(raceFile));
String read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	user = splited[0];
	race = splited[1];
	if (race.equals("African-American")) {
		protectedGroup.add(user);
		protectedNum++;
		if (truthMap.get(user))
			posProtectedNum++;
		else
			negProtectedNum++;
	} else {
		unprotectedNum++;
		if (truthMap.get(user))
			posUnprotectedNum++;
		else
			negUnprotectedNum++;
	}
}

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
		if (truthMap.get(user))
			posUnprotectedValue += value;
		else
			negUnprotectedValue += value;
	} else { // Protected
		protectedValue += value;
		if (truthMap.get(user))
			posProtectedValue += value;
		else
			negProtectedValue += value;
	}
}

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

/*
 *  AUC-ROC Per Group
 */
//List<Double> protectedValueList = new ArrayList<Double>();
//List<Double> unprotectedValueList = new ArrayList<Double>();
//List<Integer> protectedTruthList = new ArrayList<Integer>();
//List<Integer> unprotectedTruthList = new ArrayList<Integer>();
//for (GroundAtom atom : Queries.getAllAtoms(inferredDB, recid)) {
//	GroundTerm[] terms = atom.getArguments();
//	double value = atom.getValue();
//	String user = terms[0].toString()
//	if (!protectedGroup.contains(user)) { // Unprotected
//		unprotectedValueList.add(value);
//		unprotectedTruthList.add(truthMap.get(user));
//	} else { // Protected
//		protectedValueList.add(value);
//		protectedTruthList.add(truthMap.get(user));
//		
//	}
//}
//
//double[] protectedValueArr = protectedValueList.toArray();
//double[] unprotectedValueArr = unprotectedValueList.toArray();
//int[] protectedTruthArr = protectedTruthList.toArray();
//int[] unprotectedTruthArr = unprotectedTruthList.toArray();
//
//AUCcalculator calculator = new AUCcalculator();
//double protectedAUC = calculator.measure(protectedTruthArr, protectedValueArr);
//double unprotectedAUC = calculator.measure(unprotectedTruthArr, unprotectedValueArr);
//println "AUC-ROC Proteted: "+ protectedAUC
//println "AUC-ROC Unprotected: "+ unprotectedAUC


/*
 * AUC Precision-Recall Curve Per Group
 */
Partition writePart_protect = new Partition(555);
Partition writePart_unprotect = new Partition(556);
Partition truthPart_protect = new Partition(557);
Partition truthPart_unprotect = new Partition(558);

Inserter insert1 = data.getInserter(recid, writePart_protect);
Inserter insert2 = data.getInserter(recid, writePart_unprotect);
Inserter insert3 = data.getInserter(recid, truthPart_protect);
Inserter insert4 = data.getInserter(recid, truthPart_unprotect);

Set<GroundAtom> groundings = Queries.getAllAtoms(inferredDB, recid);
for (GroundAtom ga : groundings) {
	GroundTerm[] terms = ga.getArguments();
	double value = ga.getValue();
	String userName = terms[0].toString()
	if (protectedGroup.contains(userName)) {
		insert1.insertValue(value, terms)
	} else {
		insert2.insertValue(value, terms)
	}
}
groundings.clear()
groundings = Queries.getAllAtoms(wlTruthDB, recid);
for (GroundAtom ga : groundings) {
	GroundTerm[] terms = ga.getArguments();
	double value = ga.getValue();
	String userName = terms[0].toString();
	if (protectedGroup.contains(userName)) {
		insert3.insertValue(value, terms);
	} else {
		insert4.insertValue(value, terms);
	}
}
groundings.clear();

Database resultDB_protect = data.getDatabase(writePart_protect, [recid] as Set)
Database resultDB_unprotect = data.getDatabase(writePart_unprotect, [recid] as Set)
Database truthDB_protect = data.getDatabase(truthPart_protect, [recid] as Set)
Database truthDB_unprotect = data.getDatabase(truthPart_unprotect, [recid] as Set)

// Protected Group
comparator = new SimpleRankingComparator(resultDB_protect);
comparator.setBaseline(truthDB_protect);
double[] protectScore = new double[metrics.size()];
for (int r=0; r<metrics.size(); r++) {
	comparator.setRankingScore(metrics[r]);
	protectScore[r] = comparator.compare(recid);
}
// Unprotected Group
comparator = new SimpleRankingComparator(resultDB_unprotect);
comparator.setBaseline(truthDB_unprotect);
double[] unprotectScore = new double[metrics.size()];
for (int r=0; r<metrics.size(); r++) {
	comparator.setRankingScore(metrics[r]);
	unprotectScore[r] = comparator.compare(recid);
}

println "AUPRC: "+ protectScore[0]+ ", "+ unprotectScore[0]
println "NegAUPRC: "+ protectScore[1]+ ", "+ unprotectScore[1]
println "AUC-ROC: "+ protectScore[2]+ ", "+ unprotectScore[2]



