import json



class PaddleLabel:
    def __init__(self):
        # self.label = None
        pass

    def adjust(self, f):
        ajusted_lines = []
        lines = f.readlines()
        for l in lines:
            words = l.split('\t')
            img_name = words[0]
            bboxs = json.loads(words[1])
            self.adjust_bboxs(bboxs)
            ajusted_line = f'{img_name}\t{json.dumps(bboxs)}\n'
            ajusted_lines.append(ajusted_line)
        return ajusted_lines

    def adjust_bboxs(self, bboxs):
        for b in bboxs:
            points = b['points']
            x03 = min(points[0][0], points[3][0])-3
            x12 = max(points[1][0], points[2][0])+4
            y01 = min(points[0][1], points[1][1])
            y23 = max(points[2][1], points[3][1])
            b['points'] = [[x03, y01], [x12, y01], [x12, y23], [x03, y23]]







    def read(self, path='./Label.txt'):
        with open(path) as f:
            ajusted_lines = self.adjust(f)

        with open('./Label_adjusted.txt', 'w') as f:
            f.writelines(ajusted_lines)



p = PaddleLabel()
p.read()