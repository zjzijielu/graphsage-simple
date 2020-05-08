node_map = {}

with open("./citeseer.content") as fp:
    for i,line in enumerate(fp):
        info = line.strip().split()
        node_map[info[0]] = i

with open("./citeseer.cites") as fp:
    with open("./citeseer.cites.parsed", "w") as output:
        for i,line in enumerate(fp):
            info = line.strip().split()
            try:
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                output.write(str(paper1) + " " + str(paper2) + "\n")
            except:
                # print(info[0], info[1])
                continue