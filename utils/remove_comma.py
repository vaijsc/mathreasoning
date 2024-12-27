import re

def main():
    x  = "this is 100,000, USD, VND"
    template = r'(?<=\d),(?=\d)'
    print(re.sub(template, '', x))


if __name__ == "__main__":
    main()
