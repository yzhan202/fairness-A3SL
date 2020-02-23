package binghamton.test.paperReview;

import java.text.DecimalFormat;

import org.apache.commons.lang3.ArrayUtils
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import edu.umd.cs.psl.application.inference.*;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.em.HardEM
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
import edu.umd.cs.psl.model.kernel.rule.AbstractRuleKernel
import edu.umd.cs.psl.model.kernel.rule.CompatibilityRuleKernel
import edu.umd.cs.psl.model.predicate.*;
import edu.umd.cs.psl.model.term.*;
import edu.umd.cs.psl.model.rule.*;
import edu.umd.cs.psl.model.weight.*;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Term
import edu.umd.cs.psl.model.argument.UniqueID
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
import binghamton.bachuai13.util.ExperimentConfigGenerator
import binghamton.bachuai13.util.WeightLearner;
import binghamton.structureLearning.pathBasedL1_paperReview;
import binghamton.util.DataOutputter;


dataDir = 'data/biasedPaperReview/bias1';

def modelType = "quad"; //quad; bool
def methods = ["MLE"]; // MaxLikelihoodMPE

Logger log = LoggerFactory.getLogger(this.class)
ConfigManager cm = ConfigManager.getManager();
ConfigBundle cb = cm.getBundle("markovLogicNetwork");

def defPath = System.getProperty("java.io.tmpdir") + "/mln"
def dbpath = cb.getString("dbpath", defPath)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("paperreview");

configGenerator.setModelTypes([modelType]);
configGenerator.setLearningMethods(methods);
configGenerator.setVotedPerceptronStepCounts([50]);
configGenerator.setVotedPerceptronStepSizes([(double) 1.0]);

log.info("Initializing model ...");

PSLModel model = new PSLModel(this, data);

// Default
model.add predicate: "author", types: [ArgumentType.UniqueID]
model.add predicate: "paper", types: [ArgumentType.UniqueID]
model.add predicate: "reviewer", types: [ArgumentType.UniqueID]
// Feature
model.add predicate: "student", types: [ArgumentType.UniqueID]
model.add predicate: "acceptable", types: [ArgumentType.UniqueID]
model.add predicate: "reviews", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "submits", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
// Target
model.add predicate: "positiveReviews", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "positiveSummary", types: [ArgumentType.UniqueID]
// sensitive attribute
model.add predicate: "institute", types: [ArgumentType.UniqueID]
model.add predicate: "highRank", types: [ArgumentType.UniqueID]
model.add predicate: "affiliation", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

def closedPredicates = [author, paper, reviewer, student, acceptable, reviews, submits,
	institute, highRank, affiliation] as Set;
def inferredPredicates = [positiveReviews, positiveSummary] as Set; // presents

StandardPredicate[] negFeatPreds = [student, acceptable];

Set<StandardPredicate> SensitiveAttribute = [institute, affiliation, highRank] as Set;

Set<String> highRankInstitutes = new HashSet<>();
Map<String, String> instituteMap = new HashMap<>();
Set<String> studentSet = new HashSet<>();
paperAuthorMap = new HashMap<>();
protectedGroup = new HashSet<>();

//highRank
String fileName = dataDir+ '/highRank.txt';
BufferedReader reader = null;
reader = new BufferedReader(new FileReader(fileName));
String read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	int value = Integer.parseInt(splited[1]);
	if (value == 1) {
		highRankInstitutes.add(splited[0]);
	}
}
// affiliation
fileName = dataDir+ '/affiliation.txt';
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	String s = splited[0];
	String i = splited[1];
	instituteMap.put(s, i);
}
// student
fileName = dataDir+ '/student.txt';
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(',');
	String s = splited[0];
	int value = Integer.parseInt(splited[1]);
	if (value == 1)
		studentSet.add(s);
}
// submits
fileName = dataDir+ '/submits.txt';
reader = new BufferedReader(new FileReader(fileName));
read = null;
Map<String, String> paperAuthorMap = new HashMap<>();
while((read = reader.readLine()) != null) {
	String[] splited = read.split(',');
	String a = splited[0];
	String p = splited[1];
	paperAuthorMap.put(p, a);
}

int protectedNum = 0;
int unprotectedNum = 0;
Set<String> protectedGroup = new HashSet<>();
for (Map.Entry<String, String> entry : instituteMap.entrySet()) {
	String stud = entry.getKey();
	String inst = entry.getValue();
	if (studentSet.contains(stud)) {
		if (highRankInstitutes.contains(inst)) {
			unprotectedNum++; // Unprotected Group
		}
		else {
			protectedNum++; // Protected Group
			protectedGroup.add(stud);
		}
	} else {
		unprotectedNum++;
	}
}

def predicateFileMap = [
	((Predicate)author):"author.txt",
	((Predicate)paper):"paper.txt",
	((Predicate)reviewer):"reviewer.txt",
	((Predicate)student):"student.txt",
	((Predicate)acceptable):"acceptable.txt",
	((Predicate)reviews):"reviews.txt",
	((Predicate)submits):"submits.txt",
	((Predicate)positiveReviews):"positiveReviews.txt",
	((Predicate)positiveSummary):"positiveSummary.txt",
	
	((Predicate)institute):"institute.txt",
	((Predicate)highRank):"highRank.txt",
	((Predicate)affiliation):"affiliation.txt"]

def predicateSoftTruthMap = [
	((Predicate)student):true,
	((Predicate)acceptable):true,
	((Predicate)author):false,
	((Predicate)paper):false,
	((Predicate)reviewer):false,
	((Predicate)reviews):false,
	((Predicate)submits):false,
	((Predicate)positiveReviews):true,
	((Predicate)positiveSummary):true,
	
	((Predicate)institute):false,
	((Predicate)highRank):true,
	((Predicate)affiliation):false]

GenericVariable A = new GenericVariable('A', model); //author
GenericVariable P = new GenericVariable("P", model);
GenericVariable R = new GenericVariable("R", model);
GenericVariable R2 = new GenericVariable("R2", model);

GenericVariable I = new GenericVariable("I", model);

Map<StandardPredicate, List<List<GenericVariable>>> generalPredArgsMap = [
	((Predicate)author): [[A]],
	((Predicate)paper): [[P]],
	((Predicate)reviewer): [[R], [R2]],
	((Predicate)student): [[A]],
	((Predicate)acceptable): [[P]],
	((Predicate)reviews): [[R,P], [R2,P]],
	((Predicate)submits): [[A,P]],
	((Predicate)positiveReviews): [[R,P], [R2,P]],
	((Predicate)positiveSummary): [[P]],
	
	((Predicate)institute): [[I]],
	((Predicate)highRank): [[I]],
	((Predicate)affiliation): [[A,I]]
	]
	
Partition trainPart = new Partition(0);
Partition truthPart = new Partition(1);

def inserter;
for (Predicate p: closedPredicates) {
	fileName = predicateFileMap[p];
	inserter = data.getInserter(p, trainPart);
	def fullFilePath = dataDir+ '/'+ fileName;
//			println p.toString()
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

for (Predicate p: inferredPredicates) {
	fileName = predicateFileMap[p];
	inserter = data.getInserter(p, truthPart);
	def fullFilePath = dataDir + '/' + fileName;
	if(predicateSoftTruthMap[p]){
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	}
	else{
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

wlTruthDB = data.getDatabase(truthPart, inferredPredicates);

StandardPredicate[] X = closedPredicates.toArray();
StandardPredicate[] Y = inferredPredicates.toArray();

List<ConfigBundle> configs = configGenerator.getConfigs();
ConfigBundle config = configs.get(0);


def myModel = new pathBasedL1_paperReview(X, Y, model, data, wlTruthDB, trainPart, config, log,
			generalPredArgsMap, negFeatPreds, SensitiveAttribute, paperAuthorMap, protectedGroup,
			protectedNum, unprotectedNum);
		
myModel.search()



