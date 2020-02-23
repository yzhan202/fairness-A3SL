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

def fold = 4;
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
model.add predicate: "priorFelony", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "priorMisd", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "priorOther", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

// Days in Jail
model.add predicate: "longJailDay", types: [ArgumentType.UniqueID]

// Sensitive Attributes
model.add predicate: "race", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "sex", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

def closedPredicates = [user, oldAge, felony, juvFelHistory, juvMisdHistory, juvOtherHistory, 
	priors, priorFelony, priorMisd, priorOther, longJailDay, 
	race, sex] as Set; 
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
		((Predicate)priorFelony): "priorChargeFelony.txt",
		((Predicate)priorMisd): "priorChargeMisd.txt",
		((Predicate)priorOther): "priorChargeOther.txt",
		((Predicate)longJailDay): "longJailDay.txt",
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
	((Predicate)priorFelony): false,
	((Predicate)priorMisd): false,
	((Predicate)priorOther): false,
	((Predicate)longJailDay): true,
	((Predicate)race): false,
	((Predicate)sex): false]

GenericVariable U = new GenericVariable("U", model);
GenericVariable I1 = new GenericVariable("I1", model);
GenericVariable I2 = new GenericVariable("I2", model);
GenericVariable I3 = new GenericVariable("I3", model);
GenericVariable I4 = new GenericVariable("I4", model);
GenericVariable I5 = new GenericVariable("I5", model);
GenericVariable I6 = new GenericVariable("I6", model);
GenericVariable I7 = new GenericVariable("I7", model);

GenericVariable A = new GenericVariable("A", model);

generalPredArgsMap = [
	((Predicate)user): [U],
	((Predicate)recid): [U],
	((Predicate)oldAge): [U],
	((Predicate)felony): [U],
	((Predicate)juvFelHistory): [U, I1],
	((Predicate)juvMisdHistory): [U, I2],
	((Predicate)juvOtherHistory): [U, I3],
	((Predicate)priors): [U, I4],
	((Predicate)priorFelony): [U, I5],
	((Predicate)priorMisd): [U, I6],
	((Predicate)priorOther): [U, I7],
	((Predicate)longJailDay): [U],
	((Predicate)race): [U, A],
	((Predicate)sex): [U]]

/*
 * Load PSL Model
 */
boolean isPredicateString(String str) {
	if (str.contains(")") || str.contains(",") || str=="")
		return false;
	else
		return true;
}

String modelFile = 'result/compas/fold'+fold+ '/a3cModel2.txt' 
BufferedReader reader = new BufferedReader(new FileReader(modelFile));
String read = null;
while ((read = reader.readLine()) != null) {
//	println read
	FormulaContainer body = null;
	FormulaContainer head = null;
	FormulaContainer rule = null;
	
	boolean LOGIC_NOT = false;
	String[] splited = read.split(">>");
	String bodyPart = splited[0];
	String headPart = splited[1];
	String[] head_splited = (((headPart.split("\\{"))[0]).replace(" ", "")).split("\\(");
	for (int i=0; i<head_splited.length; i++) {
		if (head_splited[i].contains("~")) {
			LOGIC_NOT = true;
		} else if (isPredicateString(head_splited[i])) {
			StandardPredicate p = (StandardPredicate)PredicateFactory.getFactory().getPredicate(head_splited[i]); //
			List<GenericVariable> argsList = generalPredArgsMap.get(p);
			Object[] args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
			
			if (LOGIC_NOT) {
				head = (FormulaContainer) head.bitwiseNegate();
				LOGIC_NOT = false;
			}
		}
	}
	
	String weightPart = bodyPart.split("\\}")[0];
	bodyPart = bodyPart.split("\\}")[1].replace(" ", "");
	double weightValue = Double.parseDouble(weightPart.replace("{", ""));
	String[] body_splited = bodyPart.split("&");
	for (int i=0; i<body_splited.length; i++) {
		String[] tmp = body_splited[i].split("\\(");
		for (int j=0; j<tmp.length; j++) {
			if (tmp[j].contains("~")) {
				LOGIC_NOT = true;
			} else if (isPredicateString(tmp[j])) {
				StandardPredicate p = (StandardPredicate)PredicateFactory.getFactory().getPredicate(tmp[j]);
				List<GenericVariable> argsList = generalPredArgsMap.get(p);
				Object[] args = new Object[argsList.size()];
				args = argsList.toArray(args);
				
				if (body == null) {
					body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					if (LOGIC_NOT) {
						body = (FormulaContainer) body.bitwiseNegate();
						LOGIC_NOT = false;
					}
				} else {
					FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					if (LOGIC_NOT) {
						f_tmp = (FormulaContainer) f_tmp.bitwiseNegate();
						LOGIC_NOT = false;
					}
					body = (FormulaContainer) body.and(f_tmp);
				}
			}
		}
	}
	
	rule = (FormulaContainer) body.rightShift(head);
	Map<String, Object> argsMap = new HashMap<String, Object>();
	argsMap.put("rule", rule);
	argsMap.put("sqaured", true);
	argsMap.put("weight", weightValue);
	
	model.add(argsMap);
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
		if (truthMap.get(user)) {
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
double negProtectedValue = 0;
double posUnprotectedValue = 0;
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
		if (truthMap.get(user) ) {
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


/*
 * AUC-ROC Per Group
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
//println "NegAUPRC: "+ protectScore[1]+ ", "+ unprotectScore[1]
//println "AUC-ROC: "+ protectScore[2]+ ", "+ unprotectScore[2]



