#include <vector>
#include <time.h>
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>	
#include <Windows.h>

#include "Flock.hpp"
#include "Common.hpp"

vector<Flock> flocks;
int nb_of_flocks;

void updateEverything() {
	vector<std::thread> flock_threads;

	glClear(GL_COLOR_BUFFER_BIT);
	glClearColor(1, 1, 1, 1);

	for (int i = 0; i < nb_of_flocks; i++) {
		
		flocks[i].drawBoids();
		flock_threads.push_back(thread(&Flock::moveBoidsToNewPositions, &flocks[i]));
	}
	
	glutSwapBuffers();
	glutPostRedisplay();

	for (int i = 0; i < nb_of_flocks; i++) {

		flock_threads[i].join();
	}

	Sleep(20);
}

int main(int argc, char *argv[]) {

	srand(time(NULL));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition((1920 - width) / 2, (1080 - height) / 2);
	glutCreateWindow("Boids");
	glutDisplayFunc(updateEverything);
		

	// Initialize Positions

	// vector<Flock> flocks;

	nb_of_flocks = 7;

	for (int i = 0; i < nb_of_flocks; i++) {

		Flock flock(colors[i]);
		flocks.push_back(flock);
	}

	glutMainLoop();
}

