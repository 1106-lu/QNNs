# QNNs

This is a repository for my experiments on Quantum Neural Networks and Quantum Machine Learning via Cirq and TensorFlow
Quantum

The most prominent project is QTx a failed (by the moment) model for stochastic completion of sentences
(it's also the only part of the code that it's fully documented at the moment).

QTx:
------------------------------
It's works on the basis that phrases in language work on a stochastic manner: if you're listening to someone that says "
my name is..." probably the next word is a proper noun. If you are task to guess the next word like on the board
game [Guess Who?](https://www.google.com/search?q=guess+who%3F&safe=active&rlz=1C1GCEA_enES784ES784&sxsrf=ALeKk024lTX8CsWgIWM7h_he7S16VDHwJw:1608242537721&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjOnLGpgtbtAhVExYUKHQc-C9IQ_AUoAXoECCEQAw&biw=958&bih=920)
, and if you got for example some verbs on the table you probably would toss them out.

No we got how a stochastic way of thinking about language (although it is not very powerful) that we can recreate with
[quantum circuits](https://en.wikipedia.org/wiki/Quantum_circuit) (QCs) that behave stochastically by their own nature.
QCs consist of a series of quantum gates that are applied on a specific order, like the words on a sentences. If we
create QCs that are equivalent to sentences with the words as a quantum gates (with all the words except the last one),
we could in theory optimize them in such a way that upon measuring we get a bitstring equivalent to the last word of the
sentence.

Like in this diagram:
![Hello my name is...](https://media.discordapp.net/attachments/549524193906130944/789925326369062942/unknown.png?width=1055&height=504)
In this case the measurement of the qubits has to be a proper noun for the sentence to make sense. The actual circuit
would this one:
![Rx..](https://cdn.discordapp.com/attachments/549524193906130944/789926095642165268/unknown.png)
If we make many of these circuits and optimize them properly in theory we could create circuits never seen before
(but with gates that have gone thought the optimization), and those circuits upon measuring would give the last word of
the phrases equivalent the circuits. 