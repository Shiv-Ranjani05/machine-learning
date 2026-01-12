s = {'s','m','p','d'}
s.add('v')
print('Set after updating:',s)

s.discard('p')
print('\nSet after updating:',s)

s.remove('v')
print('\nSet after updating:',s)

print('\nPopped element', s.pop())
print('Set after updating:',s)

s.clear()
print('\nSet after updating:',s)

