import csv
with open('lrw500_detected_face_partial.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for label in ["ABOUT"]:
        for i in range(1, 51):
            writer.writerow(["{}/test/{}_{:05d}".format(label, label, i), "0"])

