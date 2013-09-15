let source = __SOURCE_DIRECTORY__

#load "NaiveBayes.fs"
open MachineLearning.NaiveBayes

open System
open System.IO
open System.Text
open System.Text.RegularExpressions

(*
First let's grab some data. The dataset is a collection
of SMS messages, marked as "Spam" or "Ham".
The original dataset has been taken from
the UC Irvine Machine Learning Repository:
http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
*)


let trainingPath = source + "\SpamTraining"
let validationPath = source + "\SpamValidation"

type Class = Spam | Ham

let spamOrHam (line: string) =
    if line.StartsWith("ham") then (Ham, line.Remove(0,4))
    elif line.StartsWith("spam") then (Spam, line.Remove(0,5))
    else failwith "What is this?"

let read path = 
    File.ReadAllLines(path)
    |> Array.map spamOrHam

let trainingSample = read trainingPath
let validationSample = read validationPath

(*
What is the probability that a SMS message 
from the training set is spam or ham?
How about the validation sample?
Proba(SMS is Spam) = count(Spam SMS) / count(SMS)
*)

let probaSpam =
    let total = Array.length trainingSample
    let spam = 
        trainingSample 
        |> Array.filter (fun x -> fst x = Spam) 
        |> Array.length
    (float)spam / (float)total

let probaHam = 1. - probaSpam

(*
What is the probability that a spam SMS message
contains the word "ringtone"? "mom"? "chat"?
Proba(Spam SMS contains "ringtone") = 
    count(Spam SMS containing "ringtone") / count(Spam SMS)
*)

let probaThatHamSpamContains (token: string) =
    let spam, ham = 
        trainingSample
        |> Array.partition (fun x -> fst x = Spam)
    let contains token (example: (Class * string)) = 
        (snd example).ToLowerInvariant().Contains(token)
    let countInSpam = 
        spam 
        |> Array.filter (contains token) 
        |> Array.length
    let countInHam =
        ham 
        |> Array.filter (contains token) 
        |> Array.length
    (float)countInHam / (float)(ham.Length),
    (float)countInSpam / (float)(spam.Length)

(*
What is the probability that a message is Spam
if it contains "chat"? "800"?
Proba(SMS is Spam if contains "chat") =
    Proba(SMS contains "chat" if it is Spam) * 
    Proba(SMS is Spam) / Proba(SMS contains "chat")
This is a direct application of Bayes' Theorem:
P(A|B) = P(B|A) x P(A) / P(B),
in this case:
P(SMS is Spam | SMS contains "chat) =
    P(SMS contains "chat" | SMS is Spam) * 
    P(SMS is Spam) / P(SMS contains "chat")
Note that we can ignore P(SMS contains "chat")
if we compare P(Spam) and P(Ham), because
P(SMS contains "chat) doesn't change the comparison.
It's not a "probability" anymore, though.
*)

let probaHamOrSpam (token: string) =
    let pInHam, pInSpam = probaThatHamSpamContains token
    let probaHam = pInHam * probaHam
    let probaSpam = pInSpam * probaSpam
    probaHam, probaSpam

(*
We can now write a classifier based on 
the presence/absence of a single feature.
*)

type Feature = string -> bool

let trainWith (sample: (Class * string) []) (feat: Feature) =
    let spam, ham = 
        trainingSample
        |> Array.partition (fun x -> fst x = Spam)

    let pHam = (float)(Array.length ham) / (float)(Array.length sample)
    let pSpam = 1. - pHam

    let hasFeature (example: (Class * string)) = 
        feat (snd example)

    let countInSpam = 
        spam 
        |> Array.filter hasFeature 
        |> Array.length
    let countInHam =
        ham 
        |> Array.filter hasFeature
        |> Array.length

    let pInHam = (float)countInHam / (float)(ham.Length)
    let pInSpam = (float)countInSpam / (float)(spam.Length)

    let classifier (text: string) =
        if feat text
        then 
            if pInHam > pInSpam then Ham else Spam
        else if pHam > pSpam then Ham else Spam

    classifier

(*
The Naive Bayes classifier uses the same idea,
but instead of using one token, it will combine
the probabilities of each token into one aggregate
probability.
Instead of coding it from scratch, we'll use then
basic implementation from NaiveBayes.fs
Below is an illustration on how to train a classifier.
*)

let testTokens = ["chat"; "800"; "mom"; "ringtone"; "prize"; "now" ] |> Set.ofList
let testClassifier = classifier bagOfWords trainingSample testTokens

validationSample.[0..19]
|> Array.iter (fun (cl, text) -> printfn "%A -> %A / %s" cl (testClassifier text) text)

