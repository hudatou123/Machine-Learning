# Q4.2 generations (max_new_tokens = 48, prefixes = hpc/prefixes3.jsonl)

Own model = final C model (outputs/final, d128 L4, val PPL 71.35). PPL of own model = PPL of its generated tokens under itself; PPL of Pythia = PPL of the generated text under Pythia-1.4B.

| Temperature | Own model PPL | Pythia-1.4B PPL |
|---|---|---|
| 0.1 | 3.45 | 5.76 |
| 0.5 | 4.62 | 9.17 |
| 1.0 | 461.45 | 19.49 |

## Temperature 0.1

### Prefix 1

> Hammer a tube called a spile into the bark of a maple tree and its translucent yellow sap will start flowing out; the tree has been tapped (pictured). This gloop drips from some 55m Canadian maple trees today, raw material for the tangy golden syrup which is one of the country’s proudest exports.

**Own model:** '\nThe first time of the year, the first time of the year, the first time of the year, the first time of the year, the first time of the year, the first time of the year, the first time of the'

**Pythia-1.4B:** '\n\nThe sap is collected in the spring, when the maple trees are still dormant, and is then boiled down to make maple syrup.\n\nThe sap is then collected in the spring, when the maple trees are still dormant, and'

### Prefix 2

> CHARLES H. MEISNER, one of the prosperous and highly-esteemed business men of Birnamwood, Shawano county, was born in Newton township, Manitowoc Co., Wis., February 14, 1860, and is a son of Frederick and Magarette (Steltzer) Meisner, both natives of Germany.

**Own model:** '\nThe first time of the year, the first time of the year, the first time of the year, the first time of the year, the first time of the year, the first time of the year, the first time of the'

**Pythia-1.4B:** ' He was educated in the public schools of Manitowoc county, and at the age of eighteen was employed in the lumber business of the town of Manitowoc, Wis., and has since been engaged in the same business in Shawano county'

### Prefix 3

> EVER WONDER WHY SOME PEOPLE CAN WORK NIGHTS AND SLEEP EASILY DURING THE DAY? A new study from Tiina Pauno, a professor at the University of Helsinki, Finland found that many night workers actually have a variant in the genetic code that makes them less influenced by Melatonin and hence the circadian rhythm.

**Own model:** '\nThe only thing is the first time to be a few years ago, the first time to be a few years ago, the first time to be a few years ago, the first time to be a few years ago, the first time'

**Pythia-1.4B:** '\n\nThe study, published in the journal PLOS ONE, found that the variant is associated with a lower risk of night shift work.\n\n“The variant is associated with a lower risk of night shift work,” Pauno said.'


## Temperature 0.5

### Prefix 1

> Hammer a tube called a spile into the bark of a maple tree and its translucent yellow sap will start flowing out; the tree has been tapped (pictured). This gloop drips from some 55m Canadian maple trees today, raw material for the tangy golden syrup which is one of the country’s proudest exports.

**Own model:** ' In the last half of the year, the whole year, the first half of the most of the most of the world, the second half of the first half of the first half of the year, the first half of the year, the'

**Pythia-1.4B:** '\n\n“It’s a big deal, and it’s been a long time coming,” says the man who started this process. “We’ve had to wait for the right weather, the right year, and the right time'

### Prefix 2

> CHARLES H. MEISNER, one of the prosperous and highly-esteemed business men of Birnamwood, Shawano county, was born in Newton township, Manitowoc Co., Wis., February 14, 1860, and is a son of Frederick and Magarette (Steltzer) Meisner, both natives of Germany.

**Own model:** '\nThe United States of America, the United States of America, the United States of America, the United States of America, the United States of America, the United States of America, the United States of America, the United States of America'

**Pythia-1.4B:** ' He was a pupil at the Shawano school, and in his early days he was employed by the Chicago, Milwaukee & St. Paul Railway. In September, 1898, he was married to Miss Lydia (Kirk) Meisner'

### Prefix 3

> EVER WONDER WHY SOME PEOPLE CAN WORK NIGHTS AND SLEEP EASILY DURING THE DAY? A new study from Tiina Pauno, a professor at the University of Helsinki, Finland found that many night workers actually have a variant in the genetic code that makes them less influenced by Melatonin and hence the circadian rhythm.

**Own model:** ' The book is the first way to be part of the moment of the other people. The book is a lot of the book, and the book is the first book, but the book is to do, but the book is a book,'

**Pythia-1.4B:** '\n\n"It is well known that the human body is not a 24-hour clock," Pauno said. "The body has an internal clock that regulates our sleep and wakefulness. The brain is the master clock and it controls our'


## Temperature 1.0

### Prefix 1

> Hammer a tube called a spile into the bark of a maple tree and its translucent yellow sap will start flowing out; the tree has been tapped (pictured). This gloop drips from some 55m Canadian maple trees today, raw material for the tangy golden syrup which is one of the country’s proudest exports.

**Own model:** ' Debt EXI-s Beverage (Reuters) makes a general figure of the sometime fruit roots and grows on earth as the bear.\nDáRelated posts:\nAs of the stylist Proceedings:\nThe Structural facilitating foodacion'

**Pythia-1.4B:** ' But the sap is never sold in the open market for immediate consumption. It takes at least a week for the sap to flow from tapped Canadian […]\n\nIn this photo illustration the sap that drips by a Canadian maple tree is taken from'

### Prefix 2

> CHARLES H. MEISNER, one of the prosperous and highly-esteemed business men of Birnamwood, Shawano county, was born in Newton township, Manitowoc Co., Wis., February 14, 1860, and is a son of Frederick and Magarette (Steltzer) Meisner, both natives of Germany.

**Own model:** ' In this post by Philip workendant of Phillip & Qual Robert’s moments of Edward was now developing sexual peoples : The U.S. was out about this.] UTPR for posts. Injection Void Leaf, 1865, His curly'

**Pythia-1.4B:** ' The family, which numbers four sons and twelve daughters, settled in the old town of Shawano in 1857, where the sons were employed in the great mills of the country, where their father was well-known for his honest endeavors'

### Prefix 3

> EVER WONDER WHY SOME PEOPLE CAN WORK NIGHTS AND SLEEP EASILY DURING THE DAY? A new study from Tiina Pauno, a professor at the University of Helsinki, Finland found that many night workers actually have a variant in the genetic code that makes them less influenced by Melatonin and hence the circadian rhythm.

**Own model:** ' European showed the ones in the African Academy of the album was argue about caseName. Should I stick again effect in new scientific OS and190 interviews for although his mum invest for the 2019 description who would "not worry about an actual reality"'

**Pythia-1.4B:** ' Instead of falling asleep in the evening, the study finds that these people are "more active and alert throughout the day." The difference is enough to be noticed without their being aware of it. But, there are also some other interesting genetic differences'
