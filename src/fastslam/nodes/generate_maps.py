#!/usr/bin/python
import drawing_tools
import sys


if __name__ == "__main__":

	try:
		first_map = int(sys.argv[1])
		last_map = int(sys.argv[2]) + 1
	except:
		print("Invalid number of maps!")
		exit(1)

	if __name__ == "__main__":
		for i in range(first_map, last_map):
			print("Generating map " + str(i) + "...")
			drawing_tools.npyToMapIMG("./maps/mp_map" + str(i) + ".npy", (40, 40), .2, 5)
		print("Done")
