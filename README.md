
# eVision-prototype
Prototype of the surround view system



# Packages

To use the Python virtual environment, you should follow the following steps:

1. Open a terminal in the main directory.
2. Make sure virtualenv package is installed:
   ```bash
   pip install virtualenv
   ```
3. Run the following command to create a virtual environment named "venv":

   ```bash
    python3.11 -m venv venv
   ```

   This will create a new directory named "venv" that contains the virtual environment.

4. Activate the virtual environment by running the following command:
   pifa

   ```bash
   .\venv\Scripts\activate
   ```

   Once activated, your terminal prompt should change to indicate that you are now working within the virtual environment.

5. Install the list of required packages/dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

6. Instal pipreqs to automatically add packages that are being used to the requirements.txt:

   ```bash
   pip install pipreqs
   ```

7. Now to add all used packages to the requirements.txt do:

   ```bash
   pipreqs . --force
   ```

8. You can now write and run your Python code within the virtual environment.

Remember to deactivate the virtual environment when you're done by running the following command:

```bash
deactivate
```
