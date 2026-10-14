import json

northeastern_university_wiki = """Northeastern University is a private research university in Boston, Massachusetts, founded in 1898 by civil engineer Frederick C. Weld and a group of local leaders who sought to create an institution focused on practical learning and professional education. The university began as an evening school for working-class students and expanded over the decades into a major research university with a global footprint.

Northeastern is especially known for its cooperative education program, or co-op, which allows students to alternate academic study with paid work experience in industry, government, and nonprofit organizations. This model became a defining feature of the university and has helped shape its connection to employers in Boston and around the world.

The university's main campus is in the Fenway and Roxbury neighborhoods of Boston, with additional campuses in Charlotte, North Carolina; San Jose, California; Seattle, Washington; and Toronto, Canada. It is home to a wide range of academic programs across engineering, computer science, business, health sciences, law, media, and the arts.

Northeastern has grown substantially in recent decades through research investment, international partnerships, and expansion into interdisciplinary graduate and professional education. The university is widely recognized for its emphasis on experiential learning and research with real-world impact.
"""

jabberwocky = """’Twas brillig, and the slithy toves
      Did gyre and gimble in the wabe:
All mimsy were the borogoves,
      And the mome raths outgrabe.

“Beware the Jabberwock, my son!
      The jaws that bite, the claws that catch!
Beware the Jubjub bird, and shun
      The frumious Bandersnatch!”

He took his vorpal sword in hand;
      Long time the manxome foe he sought—
So rested he by the Tumtum tree
      And stood awhile in thought.

And, as in uffish thought he stood,
      The Jabberwock, with eyes of flame,
Came whiffling through the tulgey wood,
      And burbled as it came!

One, two! One, two! And through and through
      The vorpal blade went snicker-snack!
He left it dead, and with its head
      He went galumphing back.

“And hast thou slain the Jabberwock?
      Come to my arms, my beamish boy!
O frabjous day! Callooh! Callay!”
      He chortled in his joy.

’Twas brillig, and the slithy toves
      Did gyre and gimble in the wabe:
All mimsy were the borogoves,
      And the mome raths outgrabe."""

examples = [northeastern_university_wiki, jabberwocky]

with open("ppl_eval_examples.jsonl", "w") as f:
  for example in examples:
    f.write(json.dumps({"document": example}) + "\n")