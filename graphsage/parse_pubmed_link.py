node_map = {}

with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
    fp.readline()
    fp.readline()
    for i, line in enumerate(fp):
        info = line.split("\t")
        node_map[info[0]] = i

with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
    with open("pubmed-data/Pubmed.cites.parsed", "w") as output:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = info[1].split(":")[1]
            paper2 = info[-1].split(":")[1]
            output.write(str(paper1) + " " + str(paper2) + "\n")

