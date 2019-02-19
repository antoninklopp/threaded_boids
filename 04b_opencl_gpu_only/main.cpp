#define _CRT_SECURE_NO_DEPRECATE

#include <vector>
#include <time.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <CL/cl.h>

#include "Flock.hpp"
#include "Common.hpp"

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)


vector<Flock> flocks;
int nb_of_flocks;

void updateEverything() {
	vector<std::thread> flock_threads;

	clock_t tStart = clock();
	struct timespec start, finish;
	double elapsed = 0;

	clock_gettime(CLOCK_MONOTONIC, &start);

	glClear(GL_COLOR_BUFFER_BIT);
	glClearColor(1, 1, 1, 1);

	for (int i = 0; i < nb_of_flocks; i++) {

		flocks[i].drawBoids();
		flocks[i].moveBoidsToNewPositions();


	}
		glutSwapBuffers();
		glutPostRedisplay();

	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	
	std::cerr << "time taken " << elapsed << endl;
}

int main(int argc, char *argv[]) {

	srand(time(NULL));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition((1920 - width) / 2, (1080 - height) / 2);
	glutCreateWindow("Boids");
	glutDisplayFunc(updateEverything);

	nb_of_flocks = 100;

	for (int i = 0; i < nb_of_flocks; i++) {

		Flock flock(colors[i]);
		flocks.push_back(flock);
	}

	glutMainLoop();
}
