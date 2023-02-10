.MODEL SMALL

.STACK 100H

.DATA    

input_arr db 20 dup(?)
output_arr db 20 dup(?) 
temp_inputCounter dw 0 
str db '#$'
linefeed db 13, 10, "$"

.CODE
MAIN PROC

;initizlize DS

MOV AX,@DATA
MOV DS,AX

; enter your code here
mov cx,20
mov si,0
mov ah,1
input:
int 21h 
cmp al,str
je end_input
mov input_arr[si],al
inc temp_inputCounter
inc si
loop input

end_input:

; new line
;mov ah, 09
;mov dx, offset linefeed
;int 21h

mov cx,temp_inputCounter 
mov ah,2 

mov si,0
mov cx,20h      ;loop1 start input[i]  i=0           
outer_loop:
mov dx,cx
mov al,input_arr[si]        ;a=input[i]   j=0    loop2start
mov di,0
mov cx,20H;temp_inputCounter
inner_loop:
mov bl,output_arr[di]         ;b=output[j]     if a==b --->loop break    else---> inc i
cmp al,bl            
je state:
inc di
loop inner_loop                 
push_array:
mov output_arr[si],al          ;output[i]=a
inc si
mov cx,dx
loop outer_loop
state:
inc si
mov cx,dx
loop outer_loop:

; new line
mov ah, 09
mov dx, offset linefeed
int 21h

mov di,0
mov cx,20H;temp_inputCounter
lastwork:
mov bl,output_arr[di]
;NOT NEEDED
cmp bl,0h
jne print:
inc di
loop lastwork:
print:
;UPTO THIS
mov dl,bl
mov ah,2
int 21h
inc di
LOOP lastwork:


end:
;exit to DOS
MOV AX,4C00H
INT 21H

MAIN ENDP
   END MAIN