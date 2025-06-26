# Aplicação de MCTS em Jogos com Informação Imperfeita

Esse projeto é uma implementação do jogo de Tarot Francês usando [Open Spiel](https://github.com/google-deepmind/open_spiel).
O objetivo é fazer agentes usando variações de MCTS (Monte Carlo Search Tree) focados em Informação Imperfeita como "Information Set" e "Multiple Observer".

## Tarot Francês

O jogo foi implementado baseado na versão descrita no [Pagat](https://www.pagat.com/tarot/frtarot.html), é um jogo de cartas do tipo _trick-taking_, onde cara turno é um truque.

** Nota: Apenas a versão para 4 jogadores foi implementada.
### Baralho

Tarot Francês é um jogo de cartas com baralho de 78 cartas, sendo: catorze cartas entre 4 **naipes**; um grupo de cartas **trunfos** de 1 a 21; mais a **Desculpa**.
Os naipes são os mesmos que um baralho comum: Copas, Espadas, Ouros e Paus. Similar ao baralho comum, vão de Às a 10, mais o Valete, Cavaleiro, Rainha e Rei (em ordem).
Dentre as cartas numeradas, apenas três se destacam: 1 _Le Petit_, 21 _Le Monde_, e 0 _Le Fool_. A Desculpa é muitas vezes considerada a carta de número 0. 
Essas três cartas são finalizadoras, _oudlers_, e determinam como será calculada a pontuação final.

### Fluxo

No início, uma pessoa é escolhiada como _dealer_ para distribuir as cartas, são 18 cartas para cada e 6 viradas para baixo, _chien_ ou cachorro.
É feita uma rodada de ofertas, cada uma contribuindo na pontuação final, as apostas são feitas no sentido anti-horário, começando a direita do _dealer_.
* _Petit_ (Pequeno)
* _Garde_ (Guarda)
* _Garde sans chien_ (Guarda sem o cachorro)
* _Garde contre chien_ (Guarda contra o cachorro)

#### Ofertas e Tomador

Após as ofertas serem feita, a oferta maior ganha e vira o _taker_, o tomador. 
Ele será o jogador inicial e os outros jogadores são defensores que não podem deixar o tomardor ganhar
Ambos _petit_ e _garde_ o jogador vira o _chien_ para cima, mostrando as 6 cartas para todos verem.
Então o tomador pode escolher cartas do _chien_ para adicionar para mão e deve descartas as mesma quantidades de cartas.
O tomador não pode discartar cartas numeradas ou Reis.
No _garde sans chien_ o tomador não vira as cartas, mas elas contam para a pontuação final como dele.
Por fim, no _garde contre chien_ o tomador não vira as cartas, e elas serão contadas como se fossem do time defensor.

#### _Poignée_ e _Chelem_

Antes de cada jogador começar a sua primeira jogada ele pode declarar duas coisas:
* _Poignée_, punhado, mostrando cartas numeradas suficientes para conseguir bônus final
* _Chelem_, declarando que o time vencerá ganhando todas as partidas

#### Truques

O tomador inicia com uma cartas, todos os outros jogadores devem seguir o **naipe** ou **trunfo**.
No caso do naipe, pode-se usar qualquer carta do mesmo naipe, porém no trunfo é obrigatório jogar um trunfo maior ou discartar qualquer carta se não tiver mais trunfos em mãos.
Ganha o truque aquele que tiver o valor do naipe ou o número do trunfo maior, aquele truque vai para a pilha do tomador ou do time defensor, dependendo de quem ganhar.

#### A Desculpa e os Contratos

A desculpa é uma escapatória caso você não tenha como seguir o naipe ou trunfo, ela não conta no truque e fica com o jogador que usou na pilha, porém o jogador deve substituir com uma carta de baixo valor como substituto.
Se a desculpa for usada no truque final existem duas possíveis regras: de acordo com Pagat fica com o time que ganhou truque, de acordo com o as regras da Federação Francêsa de Tarot, o truque vai para o time adversário.

Dependendo da quantidade de _oudlers_ o tomador tiver, ele precisa alcançar uma certa quantidade de pontos totais:
* Zero _oudlers_: 56
* Um _oudler_: 51
* Dois _oudlers_: 41
* Trê _oudlers_: 36

As cartas valem:
* Às a 10: 0.5 pontos
* Valete: 1.5 pontos
* Cavaleiro: 2.5 pontos
* Rainha: 3.5 pontos
* Rei: 4.5 pontos
* _Le Fool_ (0), _Le Petit_ (1) e _Le Monde_ (21): 4.5 pontos
* Outras cartas numeradas: 0.5 pontos

### Pontuação
A fórmula final é a seguinte:
`((25 + pt + pb) * mu) + pg + ch`

Onde **pt** é se no último truque foi usado o _le petit_, **pb** é a diferença entre a pontuação necessária para ganhar e a pontuação de truques ganho.

O **mu** é o multiplicador da oferta:
* _Petit_ 1x
* _Garde_ 2x
* _Garde sans chien_ 4x
* _Garde contre chien_ 6x

Os bônus **pg** e **ch** são o punhado e o _chelem_. O punhado se declarado vale:
* 10 trunfos: 20 pontos
* 13 trunfos: 30 pontos
* 15 trunfos: 40 pontos
O _chelem_ vale 400 pontos se declarado **e** conquistado ou 200 pontos se não declarado. Caso declarado e não conquistado, o declarador perde 200 pontos.

### Contabilizando
O jogo é do formato *zero-sum*, ou seja, a pontuação soma zero, os pontos são divididos igualmente entre o time defensor e o tomador.
Se o tomador ganha, ele deduz pontos dos outros jogadores, se o tomador perde, ele perde pontos e os defensores ganham igualmente.
